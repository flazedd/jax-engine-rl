"""Abstract Env interface.

Every env exposes `reset` and `step` as pure JAX functions with fully static
pytree shapes. The training loop `vmap`s across parallel envs and `lax.scan`s
across rollout steps; both require shape stability.
"""
from __future__ import annotations

from typing import Any, Protocol

import chex
import jax.numpy as jnp


Observation = chex.Array
Action = chex.Array
EnvState = chex.ArrayTree


class Env(Protocol):
    """Protocol every env implements.

    Implementations are plain Python objects (not flax Modules) that carry
    static config as attributes. `reset` and `step` are pure and JIT-safe.
    """

    obs_size: int
    n_actions: int

    def reset(self, key: chex.PRNGKey) -> tuple[EnvState, Observation]:
        ...

    def step(
        self,
        state: EnvState,
        action: Action,
        key: chex.PRNGKey,
    ) -> tuple[EnvState, Observation, chex.Array, chex.Array, dict[str, Any]]:
        """Return (new_state, obs, reward, done, info).

        `reward` is a scalar float array, `done` is a scalar bool array.
        `info` is a dict of auxiliary scalar arrays (may be empty).
        Shapes of every returned leaf are identical across calls.
        """
        ...


def finite_reward(reward: chex.Array) -> chex.Array:
    """Guard used by env invariants tests; returns True if all finite."""
    return jnp.all(jnp.isfinite(reward))
