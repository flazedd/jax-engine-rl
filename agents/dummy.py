"""Random-policy agent for M0 pipeline validation.

No learning. `act` samples a uniform random action and returns empty extras;
`update` is a no-op that returns constant metrics so the training loop's
shape-stable JIT path is exercised end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DummyAgent:
    obs_size: int = 1
    n_actions: int = 2

    requires_regime_label: bool = False
    requires_analytical_posterior: bool = False
    is_recurrent: bool = False
    produces_belief_for_eval: bool = False

    def init(self, key: chex.PRNGKey) -> chex.ArrayTree:
        return {"step": jnp.asarray(0, dtype=jnp.int32)}

    def act(
        self,
        state: chex.ArrayTree,
        obs: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, dict[str, chex.Array], chex.ArrayTree]:
        action = jax.random.randint(key, (), 0, self.n_actions)
        extras: dict[str, chex.Array] = {}
        return action, extras, state

    def update(
        self,
        state: chex.ArrayTree,
        trajectory: chex.ArrayTree,
        final_obs: chex.Array,
    ) -> tuple[chex.ArrayTree, dict[str, chex.Array]]:
        new_state = {**state, "step": state["step"] + 1}
        metrics = {"dummy/noop_loss": jnp.asarray(0.0, dtype=jnp.float32)}
        return new_state, metrics
