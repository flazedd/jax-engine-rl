"""Dummy env for M0 pipeline validation.

Constant reward, fixed-length episodes, 1-dim observation. No dynamics —
its sole purpose is to exercise rollout + update + plotting without any
research logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax.numpy as jnp


@dataclass(frozen=True)
class DummyEnv:
    episode_length: int = 32
    reward_value: float = 1.0
    obs_size: int = 1
    n_actions: int = 2

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        state = {"t": jnp.asarray(0, dtype=jnp.int32)}
        obs = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        return state, obs

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        t_next = state["t"] + 1
        done = t_next >= self.episode_length
        new_state = {"t": jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)}
        obs = jnp.zeros((self.obs_size,), dtype=jnp.float32)
        reward = jnp.asarray(self.reward_value, dtype=jnp.float32)
        return new_state, obs, reward, done, {}
