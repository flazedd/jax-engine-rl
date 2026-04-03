"""PPO agent implementation."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
from lob_sim.agents.base import AgentState
from lob_sim.agents.networks import MLP


class PPOConfig(NamedTuple):
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    n_envs: int = 16
    n_steps: int = 128
    hidden_size: int = 0


class PPOAgent(eqx.Module):
    trunk: MLP
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    ppo_config: PPOConfig = eqx.field(static=True)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __init__(self, ppo_config: PPOConfig = PPOConfig(), *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = MLP(33, [64, 64], 64, key=k1)
        from lob_sim.actions import N_ACTIONS
        self.policy_head = eqx.nn.Linear(64, N_ACTIONS, key=k2)
        self.value_head = eqx.nn.Linear(64, 1, key=k3)
        self.ppo_config = ppo_config

    def initial_agent_state(self, rng_key) -> AgentState:
        return AgentState(
            hidden=jnp.zeros(1),
            prev_action=jnp.int32(0),
            prev_reward=jnp.float32(0.0),
        )

    def get_action(self, obs, agent_state, rng_key):
        trunk_out = self.trunk(obs)
        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)[0]

        action = jax.random.categorical(rng_key, logits)
        log_probs = jax.nn.log_softmax(logits)
        log_prob = log_probs[action]

        return jnp.int32(action), agent_state, {"log_prob": log_prob, "value": value}
