"""StackObsEnv — concatenates the last K observations.

Wraps any env. The augmented observation is the flattened buffer of the K
most recent base observations, oldest-first. On reset, the buffer is filled
with K copies of the initial obs. On `done` (which the inner env handles via
auto-reset), the buffer is reset to K copies of the new episode's first obs
so memory does not leak across episode boundaries.

Used to give a memoryless PPO policy a fixed-length history window — the
"stacked-obs PPO" rung of the M5 method ladder. With K=1 this is a no-op
identity wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax.numpy as jnp


@dataclass(frozen=True)
class StackObsEnv:
    inner: Any  # any env implementing the Env Protocol
    k: int = 4  # window length

    @property
    def obs_size(self) -> int:
        return self.inner.obs_size * self.k

    @property
    def n_actions(self) -> int:
        return self.inner.n_actions

    @property
    def episode_length(self) -> int:
        return self.inner.episode_length

    @property
    def gamma(self) -> float:
        return getattr(self.inner, "gamma", 0.99)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        inner_state, base_obs = self.inner.reset(key)
        buffer = jnp.broadcast_to(
            base_obs, (self.k,) + base_obs.shape
        ).astype(base_obs.dtype)
        state = {"inner": inner_state, "buffer": buffer}
        return state, buffer.reshape(-1)

    def step(
        self, state: chex.ArrayTree, action: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        new_inner, base_obs, reward, done, info = self.inner.step(
            state["inner"], action, key
        )
        shifted = jnp.concatenate(
            [state["buffer"][1:], base_obs[None]], axis=0
        )
        fresh = jnp.broadcast_to(base_obs, state["buffer"].shape).astype(
            state["buffer"].dtype
        )
        new_buffer = jnp.where(done, fresh, shifted)
        new_state = {"inner": new_inner, "buffer": new_buffer}
        return new_state, new_buffer.reshape(-1), reward, done, info
