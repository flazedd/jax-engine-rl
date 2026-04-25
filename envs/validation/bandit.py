"""Two-armed Bernoulli bandit meta-task (validation env for M4).

Per-episode arm probabilities are drawn IID from `Uniform[0, 1]`. The agent
acts for `episode_length` steps; reward is `Bernoulli(p[action])`. The
observation is a constant zero vector — within-episode signal arrives at
recurrent / variational agents through (prev_action, prev_reward) wrappers,
not through the env channel.

This is the canonical RL² / VariBAD validation task. Vanilla feedforward PPO
must plateau near the random-arm baseline (~0.5 × episode_length) because
obs carries no state. RL² and VariBAD should climb above this floor toward
the Bayes-optimal ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BanditEnv:
    n_arms: int = 2
    episode_length: int = 10

    @property
    def obs_size(self) -> int:
        return 1

    @property
    def n_actions(self) -> int:
        return self.n_arms

    def _zero_obs(self) -> chex.Array:
        return jnp.zeros((self.obs_size,), dtype=jnp.float32)

    def _sample_arm_probs(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(
            key, (self.n_arms,), dtype=jnp.float32, minval=0.0, maxval=1.0
        )

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        arm_probs = self._sample_arm_probs(key)
        state = {
            "arm_probs": arm_probs,
            "t": jnp.asarray(0, dtype=jnp.int32),
        }
        return state, self._zero_obs()

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        arm_probs = state["arm_probs"]
        action = action.astype(jnp.int32)
        p = arm_probs[action]

        k_reward, k_reset = jax.random.split(key)
        u = jax.random.uniform(k_reward, ())
        reward = (u < p).astype(jnp.float32)

        t_next = state["t"] + 1
        done = t_next >= self.episode_length

        # Always compute both branches; pick via jnp.where to keep shapes static.
        new_arm_probs = jnp.where(
            done, self._sample_arm_probs(k_reset), arm_probs
        )
        t_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)

        new_state = {"arm_probs": new_arm_probs, "t": t_after}
        info = {
            "arm_probs": arm_probs,
            "best_arm": jnp.argmax(arm_probs).astype(jnp.int32),
            "best_arm_prob": jnp.max(arm_probs),
            "regret_step": jnp.max(arm_probs) - p,
        }
        return new_state, self._zero_obs(), reward, done, info
