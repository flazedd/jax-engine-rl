"""Chain MDP — JAX meta-RL validation environment.

N-state chain with hidden goal. Agent must explore to find the goal,
then navigate to it and stay. Tests state-dependent exploration and
memory — harder than the bandit since actions affect future states.

States: 0, 1, ..., N-1 arranged in a chain.
Actions: 0=left, 1=stay, 2=right (deterministic transitions, clipped).
Goal: sampled uniformly per episode (hidden from agent).
Reward: +1 when at goal state, 0 otherwise.
Observation: one_hot(state, N) — agent sees its position but not the goal.

Interface matches lob_sim.jax_env:
    env_reset(key, params) -> (state, obs)
    env_step(key, state, action, params) -> (state, obs, reward, done, info)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class ChainParams(NamedTuple):
    """Immutable chain MDP parameters."""
    n_states: int = 5
    t_episode: int = 20
    n_actions: int = 3   # left=0, stay=1, right=2


class ChainState(NamedTuple):
    """Mutable chain MDP state."""
    pos: jnp.ndarray       # int32 scalar — current position
    goal: jnp.ndarray      # int32 scalar — hidden goal state
    step: jnp.ndarray      # int32 scalar


def env_reset(key, params):
    """Sample new goal and start at random position."""
    k_pos, k_goal = jax.random.split(key)
    pos = jax.random.randint(k_pos, (), 0, params.n_states)
    goal = jax.random.randint(k_goal, (), 0, params.n_states)
    state = ChainState(
        pos=jnp.int32(pos), goal=jnp.int32(goal), step=jnp.int32(0))
    obs = jax.nn.one_hot(pos, params.n_states)
    return state, obs


def env_step(key, state, action, params):
    """Take action, receive reward, advance step."""
    # Deterministic transitions: left=-1, stay=0, right=+1
    delta = action - 1  # 0->-1, 1->0, 2->+1
    new_pos = jnp.clip(state.pos + delta, 0, params.n_states - 1)
    new_pos = jnp.int32(new_pos)

    reward = (new_pos == state.goal).astype(jnp.float32)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)

    new_state = ChainState(pos=new_pos, goal=state.goal, step=new_step)
    obs = jax.nn.one_hot(new_pos, params.n_states)
    return new_state, obs, reward, done, jnp.int32(0)
