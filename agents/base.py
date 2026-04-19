"""Abstract Agent interface.

Every agent:
- is constructed with env-specific sizes (`obs_size`, `n_actions`) so the
  internal model can be built without deferring to `init`;
- exposes `init(key) → AgentState` (pytree of params, opt_state, hidden state);
- exposes `act(state, obs, key) → (action, extras, state)` where `extras` is
  a pytree of per-step auxiliary outputs (e.g. `log_prob`, `value` for PPO,
  hidden-state snapshots for recurrent agents);
- exposes `update(state, trajectory, final_obs) → (state, metrics)` where
  `trajectory` contains `obs, action, reward, done, extras_*` stacked along
  time, and `final_obs` is the per-env observation just after the rollout ends
  (needed for bootstrap value computation in GAE).

`AgentState` is a JAX-compatible pytree with static shapes across calls.
"""
from __future__ import annotations

from typing import Any, Protocol

import chex


AgentState = chex.ArrayTree


class Agent(Protocol):
    obs_size: int
    n_actions: int

    requires_regime_label: bool = False
    requires_analytical_posterior: bool = False
    is_recurrent: bool = False
    produces_belief_for_eval: bool = False

    def init(self, key: chex.PRNGKey) -> AgentState:
        ...

    def act(
        self,
        state: AgentState,
        obs: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.Array, chex.ArrayTree, AgentState]:
        ...

    def update(
        self,
        state: AgentState,
        trajectory: chex.ArrayTree,
        final_obs: chex.Array,
    ) -> tuple[AgentState, dict[str, chex.Array]]:
        ...
