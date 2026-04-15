"""Contextual Bandit with Side Information — task-dependent Gaussian obs.

K-armed bandit where each episode has a hidden task (which arm is best).
In addition to the standard reward signal, the agent receives a noisy 2D
side signal drawn i.i.d. from a task-dependent Gaussian at every step.

VariBAD's observation-prediction decoder can exploit the side signals for
dense supervised gradients on task identity.  RL² agents must discover the
side signal's value purely through RL gradients — harder credit assignment.

Interface matches envs/bandit.py:
    env_reset(key, params) -> (state, obs)
    env_step(key, state, action, params) -> (state, obs, reward, done, info)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# Side-signal cluster centres in 4D (one per task/arm)
# Min pairwise distance = 2.0; well-separated for 5 tasks
SIDE_MEANS = jnp.array([
    [+1.0, +1.0,  0.0,    0.0],     # task 0 → arm 0 best
    [+1.0, -1.0,  0.0,    0.0],     # task 1 → arm 1 best
    [-1.0,  0.0, +1.0,    0.0],     # task 2 → arm 2 best
    [-1.0,  0.0, -1.0,    0.0],     # task 3 → arm 3 best
    [ 0.0,  0.0,  0.0,  1.414],     # task 4 → arm 4 best
])


class BanditContextParams(NamedTuple):
    """Immutable contextual bandit parameters."""
    n_arms: int = 5
    t_episode: int = 15
    side_dim: int = 4
    side_sigma: float = 2.5       # noisy: single obs ≈ chance level
    arm_prob_high: float = 0.7
    arm_prob_low: float = 0.35


class BanditContextState(NamedTuple):
    """Mutable bandit state."""
    arm_probs: jnp.ndarray        # (n_arms,)
    task_id: jnp.ndarray          # int32 scalar
    step: jnp.ndarray             # int32 scalar


def env_reset(key, params):
    """Sample task, set arm probs, emit initial side-signal obs."""
    k_task, k_side = jax.random.split(key)
    task_id = jax.random.randint(k_task, (), 0, params.n_arms)
    arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
    arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
    state = BanditContextState(arm_probs=arm_probs, task_id=task_id,
                               step=jnp.int32(0))
    side = SIDE_MEANS[task_id] + params.side_sigma * jax.random.normal(
        k_side, (params.side_dim,))
    obs = jnp.concatenate([jnp.zeros(1), side])   # (time=0, side_x, side_y)
    return state, obs


def env_step(key, state, action, params):
    """Pull arm `action`, receive Bernoulli reward + fresh side signal."""
    k_rew, k_side = jax.random.split(key)
    reward = jax.random.bernoulli(
        k_rew, state.arm_probs[action]).astype(jnp.float32)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = BanditContextState(
        arm_probs=state.arm_probs, task_id=state.task_id, step=new_step)
    side = SIDE_MEANS[state.task_id] + params.side_sigma * jax.random.normal(
        k_side, (params.side_dim,))
    time_sig = new_step.astype(jnp.float32) / params.t_episode
    obs = jnp.concatenate([time_sig[None], side])
    return new_state, obs, reward, done, jnp.int32(0)
