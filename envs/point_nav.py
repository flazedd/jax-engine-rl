"""2D Point Navigation with hidden goal (VariBAD semi-circle task, discretized).

Classic meta-RL dummy task (Zintgraf et al. 2019). Agent starts at origin
and must navigate to a goal sampled uniformly on the upper semicircle of
radius GOAL_RADIUS. Obs is position only — goal is hidden. Reward is +1 at
the goal (within ε), 0 otherwise. Meta-learning agents infer goal location
from past reward signals within the trial.

Interface matches envs/bandit.py:
    env_reset(key, params) -> (state, obs)
    env_step(key, state, action, params) -> (state, obs, reward, done, info)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# 4-compass actions + stay
ACTION_DELTAS = jnp.array([
    [0.0, 1.0],    # 0: up
    [0.0, -1.0],   # 1: down
    [1.0, 0.0],    # 2: right
    [-1.0, 0.0],   # 3: left
    [0.0, 0.0],    # 4: stay
])

N_ACTIONS = 5
OBS_SIZE = 2
GOAL_RADIUS = 0.8
STEP_SIZE = 0.1
GOAL_TOLERANCE = 0.15


class PointNavParams(NamedTuple):
    n_actions: int = N_ACTIONS
    t_episode: int = 50
    step_size: float = STEP_SIZE
    goal_radius: float = GOAL_RADIUS
    goal_tolerance: float = GOAL_TOLERANCE
    bound: float = 1.2          # world clipped to [-bound, bound]^2


class PointNavState(NamedTuple):
    pos: jnp.ndarray            # (2,) float
    goal: jnp.ndarray           # (2,) float — hidden from agent
    step: jnp.ndarray           # int32
    task_id: jnp.ndarray        # float angle, in [0, π]; kept for probe compat


def _sample_goal(key, params):
    angle = jax.random.uniform(key, (), minval=0.0, maxval=jnp.pi)
    return params.goal_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)]), angle


def env_reset(key, params):
    k_goal, _ = jax.random.split(key)
    goal, angle = _sample_goal(k_goal, params)
    state = PointNavState(
        pos=jnp.zeros(2),
        goal=goal,
        step=jnp.int32(0),
        task_id=angle)
    return state, state.pos


def env_step(key, state, action, params):
    # Discrete action index → delta
    safe_a = jnp.clip(action, 0, params.n_actions - 1)
    delta = ACTION_DELTAS[safe_a] * params.step_size
    new_pos = jnp.clip(state.pos + delta, -params.bound, params.bound)
    dist = jnp.linalg.norm(new_pos - state.goal)
    at_goal = dist < params.goal_tolerance
    reward = at_goal.astype(jnp.float32)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = PointNavState(
        pos=new_pos, goal=state.goal, step=new_step, task_id=state.task_id)
    return new_state, new_pos, reward, done, jnp.int32(0)
