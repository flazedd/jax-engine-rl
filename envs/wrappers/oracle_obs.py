"""OracleObsEnv — appends the true regime one-hot to the observation.

Wraps any regime-switching env that exposes `n_regimes` and stores the
true regime under `state["regime"]`. Used only by Oracle-PPO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class OracleObsEnv:
    inner: Any  # any env exposing n_regimes and state["regime"]

    @property
    def n_inventory_states(self) -> int:
        return self.inner.n_inventory_states

    @property
    def obs_size(self) -> int:
        return self.inner.obs_size + max(1, self.inner.n_regimes)

    @property
    def n_actions(self) -> int:
        return self.inner.n_actions

    @property
    def episode_length(self) -> int:
        return self.inner.episode_length

    @property
    def gamma(self) -> float:
        return getattr(self.inner, "gamma", 0.99)

    def _augment(self, base_obs: chex.Array, regime: chex.Array) -> chex.Array:
        n_reg = max(1, self.inner.n_regimes)
        one_hot = jax.nn.one_hot(regime, n_reg, dtype=jnp.float32)
        return jnp.concatenate([base_obs, one_hot], axis=-1)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        state, base_obs = self.inner.reset(key)
        obs = self._augment(base_obs, state["regime"])
        return state, obs

    def step(
        self, state: chex.ArrayTree, action: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        new_state, base_obs, reward, done, info = self.inner.step(state, action, key)
        obs = self._augment(base_obs, new_state["regime"])
        return new_state, obs, reward, done, info
