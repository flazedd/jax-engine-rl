"""Vanilla PPO (MLP actor-critic).

Used directly as the regime-agnostic floor and, wrapped in a stack_obs
observation wrapper, as the stacked-obs PPO ladder rung. Oracle-PPO and
Belief-PPO subclass this by augmenting the observation with additional
inputs (true regime / analytical posterior) — the loss and update are the
same PPO machinery from `training/ppo_update.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from training.ppo_update import compute_gae, run_ppo_epochs


class ActorCritic(nn.Module):
    n_actions: int
    hidden_dim: int = 64
    tanh_activation: bool = True
    # Optional extra trunk layers appended after the default 2-layer body
    # at `policy_trunk_hidden` width. Used by the stacked-obs cell of the
    # matched-compute factorial to reach ~180k parameters. Defaults of 0
    # reproduce the historical 2-layer-at-hidden_dim architecture.
    policy_trunk_layers: int = 0
    policy_trunk_hidden: int = 0

    @nn.compact
    def __call__(self, obs: chex.Array) -> tuple[chex.Array, chex.Array]:
        act = nn.tanh if self.tanh_activation else nn.relu
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(obs)
        x = act(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = act(x)
        for _ in range(self.policy_trunk_layers):
            x = nn.Dense(
                self.policy_trunk_hidden,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
            )(x)
            x = act(x)
        logits = nn.Dense(
            self.n_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)
        return logits, value


@dataclass(frozen=True)
class PPOAgent:
    obs_size: int
    n_actions: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    epochs: int = 4
    minibatch_size: int = 256
    # Environment-wise minibatching, matching the scheme the recurrent agents
    # are forced into. When set, minibatches are whole environment columns of
    # `minibatch_envs` environments rather than `minibatch_size` transitions
    # drawn freely across time and environments. This equalises the update rule
    # across the method ladder: the same number of optimiser steps per training
    # iteration, on the same number of transitions each, with the same
    # composition. Left at 0 the agent keeps the sample-wise scheme.
    minibatch_envs: int = 0

    requires_regime_label: bool = False
    requires_analytical_posterior: bool = False
    is_recurrent: bool = False
    produces_belief_for_eval: bool = False

    # ---- model factory --------------------------------------------------

    def _model(self) -> ActorCritic:
        return ActorCritic(n_actions=self.n_actions, hidden_dim=self.hidden_dim)

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate),
        )

    # ---- Agent API ------------------------------------------------------

    def init(self, key: chex.PRNGKey) -> chex.ArrayTree:
        model = self._model()
        dummy_obs = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        params = model.init(key, dummy_obs)
        opt_state = self._optimizer().init(params)
        return {
            "params": params,
            "opt_state": opt_state,
            "update_key": jax.random.PRNGKey(0),
        }

    def act(
        self,
        state: chex.ArrayTree,
        obs: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, dict[str, chex.Array], chex.ArrayTree]:
        logits, value = self._model().apply(state["params"], obs)
        action = jax.random.categorical(key, logits)
        log_prob = jax.nn.log_softmax(logits)[action]
        extras = {"log_prob": log_prob, "value": value}
        return action, extras, state

    def update(
        self,
        state: chex.ArrayTree,
        trajectory: chex.ArrayTree,
        final_obs: chex.Array,
    ) -> tuple[chex.ArrayTree, dict[str, chex.Array]]:
        params = state["params"]
        opt_state = state["opt_state"]

        # Bootstrap value for the final observation (per env).
        _, last_value = jax.vmap(lambda o: self._model().apply(params, o))(final_obs)

        advantages, returns = compute_gae(
            trajectory["reward"],
            trajectory["value"],
            trajectory["done"],
            last_value,
            gamma=self.gamma,
            lam=self.lam,
        )

        # Flatten [T, N, ...] → [T*N, ...] for minibatching.
        def _flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        batch = {
            "obs": _flatten(trajectory["obs"]),
            "action": _flatten(trajectory["action"]),
            "log_prob": _flatten(trajectory["log_prob"]),
            "advantage": _flatten(advantages),
            "return": _flatten(returns),
        }

        update_key, next_key = jax.random.split(state["update_key"])
        if self.minibatch_envs > 0:
            # Group the flattened batch into whole environment columns so a
            # minibatch is `minibatch_envs` complete trajectories, the scheme
            # the recurrent agents are forced into. `_flatten` maps (t, n) to
            # t*N + n, so environment n owns rows n, n+N, n+2N, ...; gathering
            # in that order puts each environment's T steps contiguous.
            # `shuffle_block` then keeps the per-epoch reshuffle at trajectory
            # granularity instead of breaking the columns apart again.
            n_steps, n_env = trajectory["reward"].shape[:2]
            idx = (
                jnp.arange(n_env)[:, None] + jnp.arange(n_steps)[None, :] * n_env
            ).reshape(-1)
            batch = jax.tree_util.tree_map(lambda x: x[idx], batch)
            # Cap to the environments actually present, as the recurrent agents
            # do: a reduced-scale run can have fewer environments than the
            # configured minibatch, which would otherwise yield no minibatches.
            mb_envs = min(self.minibatch_envs, n_env)
            minibatch_size = mb_envs * n_steps
            shuffle_block = n_steps
        else:
            minibatch_size = self.minibatch_size
            shuffle_block = 0
        params, opt_state, metrics = run_ppo_epochs(
            lambda p, o: self._model().apply(p, o),
            self._optimizer(),
            params,
            opt_state,
            batch,
            key=update_key,
            epochs=self.epochs,
            minibatch_size=minibatch_size,
            shuffle_block=shuffle_block,
            clip_eps=self.clip_eps,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
        )

        new_state = {
            **state,
            "params": params,
            "opt_state": opt_state,
            "update_key": next_key,
        }
        metrics["ppo/advantage_mean"] = advantages.mean()
        metrics["ppo/return_mean"] = returns.mean()
        return new_state, metrics

    # ---- diagnostics ----------------------------------------------------

    def action_probs(self, state: chex.ArrayTree, obs: chex.Array) -> chex.Array:
        """Categorical probability over actions for a single obs. JIT-friendly."""
        logits, _ = self._model().apply(state["params"], obs)
        return jax.nn.softmax(logits)
