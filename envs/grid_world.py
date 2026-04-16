"""Grid World — goal navigation with hidden task (VariBAD GridNavi).

5×5 grid. Agent starts at center (2,2). Goal is sampled from 8
fixed positions (corners + edge midpoints). Goal is hidden — the
agent observes only its (x, y) position as normalized coordinates.

Actions: up(0), down(1), left(2), right(3), stay(4).
Reward: +1 per step at goal, 0 otherwise.
Episode: T=15 steps.

In multi-episode trials, the goal is fixed across all episodes — the
agent must explore to find the goal in early episodes and exploit in
later episodes. Use `env_episode_reset` for within-trial episode
boundaries (keeps goal, resets position to center).

Interface matches envs/chain.py:
    env_reset(key, params) -> (state, obs)
    env_step(key, state, action, params) -> (state, obs, reward, done, info)
    env_episode_reset(state, params) -> (state, obs)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Goal positions: corners + edge midpoints of 5×5 grid
GOAL_POSITIONS = jnp.array([
    [0, 0], [0, 4], [4, 0], [4, 4],   # corners
    [0, 2], [2, 0], [4, 2], [2, 4],   # edge midpoints
], dtype=jnp.int32)

N_GOALS = 8

# Movement deltas per action
# up(0): y-1, down(1): y+1, left(2): x-1, right(3): x+1, stay(4): 0
DX = jnp.array([0, 0, -1, 1, 0], dtype=jnp.int32)
DY = jnp.array([-1, 1, 0, 0, 0], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# State / Params
# ---------------------------------------------------------------------------

class GridWorldParams(NamedTuple):
    grid_size: int = 5
    n_goals: int = N_GOALS
    t_episode: int = 15
    n_actions: int = 5     # up, down, left, right, stay


class GridWorldState(NamedTuple):
    pos_x: jnp.ndarray      # int32 scalar
    pos_y: jnp.ndarray      # int32 scalar
    goal_idx: jnp.ndarray   # int32 scalar — index into GOAL_POSITIONS
    step: jnp.ndarray       # int32 scalar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(pos_x, pos_y, grid_size):
    """Observation: normalized (x, y) in [0, 1]²."""
    return jnp.array([pos_x / (grid_size - 1),
                      pos_y / (grid_size - 1)], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Core env functions
# ---------------------------------------------------------------------------

def env_reset(key, params):
    """Full reset: sample new goal, place agent at center."""
    goal_idx = jax.random.randint(key, (), 0, params.n_goals)
    cx = jnp.int32(params.grid_size // 2)
    cy = jnp.int32(params.grid_size // 2)
    state = GridWorldState(
        pos_x=cx, pos_y=cy,
        goal_idx=jnp.int32(goal_idx), step=jnp.int32(0))
    obs = _make_obs(cx, cy, params.grid_size)
    return state, obs


def env_episode_reset(state, params):
    """Within-trial reset: keep goal, reset position to center."""
    cx = jnp.int32(params.grid_size // 2)
    cy = jnp.int32(params.grid_size // 2)
    new_state = GridWorldState(
        pos_x=cx, pos_y=cy,
        goal_idx=state.goal_idx, step=jnp.int32(0))
    obs = _make_obs(cx, cy, params.grid_size)
    return new_state, obs


def env_step(key, state, action, params):
    """Take action, receive reward. Key unused (deterministic transitions)."""
    new_x = jnp.clip(state.pos_x + DX[action], 0, params.grid_size - 1)
    new_y = jnp.clip(state.pos_y + DY[action], 0, params.grid_size - 1)

    goal = GOAL_POSITIONS[state.goal_idx]
    at_goal = (new_x == goal[0]) & (new_y == goal[1])
    reward = jnp.where(at_goal, 1.0, 0.0)

    new_step = state.step + 1
    done = (new_step >= params.t_episode)

    new_state = GridWorldState(
        pos_x=jnp.int32(new_x), pos_y=jnp.int32(new_y),
        goal_idx=state.goal_idx, step=jnp.int32(new_step))
    obs = _make_obs(new_x, new_y, params.grid_size)
    return new_state, obs, reward, done, state.goal_idx
