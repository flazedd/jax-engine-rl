"""Agent types and protocol."""
import typing
from typing import Any, NamedTuple

import jax.numpy as jnp


class AgentState(NamedTuple):
    """Carries agent-internal state across timesteps inside lax.scan.
    For stateless agents (PPO), all fields are dummy zeros.
    For recurrent agents (RL2), hidden holds the GRU hidden state.
    """
    hidden: jnp.ndarray
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray
    prev_done: jnp.ndarray = jnp.float32(0.0)


class RolloutBatch(NamedTuple):
    """One batch of experience collected by collect_rollout()."""
    obs: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    last_value: jnp.ndarray


class Agent(typing.Protocol):
    def initial_agent_state(self, rng_key: jnp.ndarray) -> AgentState: ...

    def get_action(
        self,
        obs: jnp.ndarray,
        agent_state: AgentState,
        rng_key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, AgentState, dict]:
        """Returns (action_scalar_int32, new_agent_state, info).
        info must contain:
          'log_prob': scalar float32
          'value':    scalar float32
        """
        ...

    def update(
        self,
        batch: RolloutBatch,
        opt_state: Any,
        rng_key: jnp.ndarray,
    ) -> tuple["Agent", Any, dict]:
        """Returns (updated_agent, updated_opt_state, metrics_dict)."""
        ...
