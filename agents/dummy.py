"""Random-policy agent for M0 pipeline validation.

No learning. `act` samples a uniform random action; `update` is a no-op that
returns constant metrics so the training loop's shape-stable JIT path is
exercised end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DummyAgent:
    requires_regime_label: bool = False
    requires_analytical_posterior: bool = False
    is_recurrent: bool = False
    produces_belief_for_eval: bool = False

    def init(
        self,
        key: chex.PRNGKey,
        obs_size: int,
        n_actions: int,
        config: dict[str, Any],
    ) -> chex.ArrayTree:
        return {
            "n_actions": jnp.asarray(n_actions, dtype=jnp.int32),
            "step": jnp.asarray(0, dtype=jnp.int32),
        }

    def act(
        self,
        state: chex.ArrayTree,
        obs: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.ArrayTree]:
        n_actions = state["n_actions"]
        action = jax.random.randint(key, (), 0, n_actions)
        return action, state

    def update(
        self,
        state: chex.ArrayTree,
        trajectory_batch: chex.ArrayTree,
    ) -> tuple[chex.ArrayTree, dict[str, chex.Array]]:
        new_state = {**state, "step": state["step"] + 1}
        metrics = {
            "dummy/noop_loss": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return new_state, metrics
