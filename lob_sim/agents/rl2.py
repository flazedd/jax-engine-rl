"""RL² agent — GRU-based recurrent agent for meta-learning."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from lob_sim.agents.base import AgentState
from lob_sim.agents.networks import GRUCell
from lob_sim.actions import N_ACTIONS


class RL2Config(NamedTuple):
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    n_envs: int = 64
    n_steps: int = 512
    hidden_size: int = 128
    n_episodes_per_meta: int = 3


class RL2Agent(eqx.Module):
    gru: GRUCell
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __init__(self, hidden_size: int = 128, *, key):
        self.hidden_size = hidden_size
        self.n_actions = N_ACTIONS
        k1, k2, k3 = jax.random.split(key, 3)
        # Input: obs(33) + one-hot action(N_ACTIONS) + prev_reward(1) + prev_done(1)
        input_size = 33 + N_ACTIONS + 1 + 1
        self.gru = GRUCell(input_size, hidden_size, key=k1)
        self.policy_head = eqx.nn.Linear(hidden_size, N_ACTIONS, key=k2)
        self.value_head = eqx.nn.Linear(hidden_size, 1, key=k3)

    def initial_agent_state(self, rng_key) -> AgentState:
        return AgentState(
            hidden=jnp.zeros(self.hidden_size),
            prev_action=jnp.int32(0),
            prev_reward=jnp.float32(0.0),
            prev_done=jnp.float32(0.0),
        )

    def get_action(self, obs, agent_state, rng_key):
        # Build GRU input: [obs, one_hot(prev_action), prev_reward, prev_done]
        prev_action_onehot = jax.nn.one_hot(agent_state.prev_action, self.n_actions)
        gru_input = jnp.concatenate([
            obs,
            prev_action_onehot,
            jnp.atleast_1d(agent_state.prev_reward),
            jnp.atleast_1d(agent_state.prev_done),
        ])

        # GRU step
        new_hidden = self.gru(gru_input, agent_state.hidden)

        # Policy and value heads
        logits = self.policy_head(new_hidden)
        value = self.value_head(new_hidden)[0]

        # Sample action
        action = jax.random.categorical(rng_key, logits)
        log_probs = jax.nn.log_softmax(logits)
        log_prob = log_probs[action]

        new_state = AgentState(
            hidden=new_hidden,
            prev_action=jnp.int32(action),
            prev_reward=agent_state.prev_reward,  # updated by rollout
            prev_done=agent_state.prev_done,       # updated by rollout
        )

        return jnp.int32(action), new_state, {"log_prob": log_prob, "value": value}
