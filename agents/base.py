"""Abstract Agent interface.

Every agent implements init / act / update. `AgentState` is a pytree carrying
params, optimizer state, and any recurrent hidden state — JAX-compatible with
static shapes across calls so the training loop compiles once per seed.
"""
from __future__ import annotations

from typing import Any, Protocol

import chex


AgentState = chex.ArrayTree


class Agent(Protocol):
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
    ) -> AgentState:
        ...

    def act(
        self,
        state: AgentState,
        obs: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, AgentState]:
        ...

    def update(
        self,
        state: AgentState,
        trajectory_batch: chex.ArrayTree,
    ) -> tuple[AgentState, dict[str, chex.Array]]:
        ...
