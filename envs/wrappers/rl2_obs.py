"""RL² observation wrapper.

Augments the base observation with the previous step's `(action, reward, done)`
so a feedforward-or-recurrent policy can see its own history. RL² uses a GRU
on top of this augmented obs; the GRU's hidden state plus the augmented obs
together form the agent's implicit belief.

Augmented obs layout: `[base_obs, prev_action_one_hot, prev_reward, prev_done]`.
At reset, all "prev" fields are zero (no prior history).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RL2ObsEnv:
    inner: Any  # any env implementing the Env Protocol

    @property
    def obs_size(self) -> int:
        return self.inner.obs_size + self.inner.n_actions + 2

    @property
    def n_actions(self) -> int:
        return self.inner.n_actions

    @property
    def episode_length(self) -> int:
        return self.inner.episode_length

    def _augment(
        self,
        base_obs: chex.Array,
        prev_action: chex.Array,
        prev_reward: chex.Array,
        prev_done: chex.Array,
    ) -> chex.Array:
        a_one_hot = jax.nn.one_hot(prev_action, self.inner.n_actions, dtype=jnp.float32)
        return jnp.concatenate(
            [
                base_obs,
                a_one_hot,
                prev_reward[None].astype(jnp.float32),
                prev_done[None].astype(jnp.float32),
            ],
            axis=-1,
        )

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        inner_state, base_obs = self.inner.reset(key)
        prev_action = jnp.asarray(0, dtype=jnp.int32)
        prev_reward = jnp.asarray(0.0, dtype=jnp.float32)
        prev_done = jnp.asarray(0.0, dtype=jnp.float32)
        state = {
            "inner": inner_state,
            "prev_action": prev_action,
            "prev_reward": prev_reward,
            "prev_done": prev_done,
        }
        return state, self._augment(base_obs, prev_action, prev_reward, prev_done)

    def step(
        self, state: chex.ArrayTree, action: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        new_inner, base_obs, reward, done, info = self.inner.step(
            state["inner"], action, key
        )
        # When done, the inner env has already auto-reset its state; the
        # next obs is the start of a new episode. Zero the "prev" fields
        # so the new episode sees a fresh history. Otherwise, pass through
        # this step's (action, reward, done).
        zero_prev_action = jnp.asarray(0, dtype=jnp.int32)
        zero_prev_reward = jnp.asarray(0.0, dtype=jnp.float32)
        prev_action_after = jnp.where(done, zero_prev_action, action.astype(jnp.int32))
        prev_reward_after = jnp.where(done, zero_prev_reward, reward.astype(jnp.float32))
        prev_done_after = done.astype(jnp.float32)

        new_state = {
            "inner": new_inner,
            "prev_action": prev_action_after,
            "prev_reward": prev_reward_after,
            "prev_done": prev_done_after,
        }
        obs = self._augment(
            base_obs, prev_action_after, prev_reward_after, prev_done_after
        )
        return new_state, obs, reward, done, info
