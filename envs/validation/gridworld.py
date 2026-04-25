"""Random-goal gridworld meta-task (validation env for M4).

A 5×5 grid; agent starts at the center cell. Each episode draws a goal cell
uniformly from the 24 non-center cells. Reward is `1` on any step the agent
sits on the goal, `0` otherwise (a "lingering" goal — once found, an agent
that stays put keeps collecting). Observation is the agent's normalized (x,y)
position in `[0, 1]²`; the goal is hidden.

This is the canonical RL² / VariBAD spatial-meta-task. Vanilla feedforward PPO
plateaus at the random-search baseline because obs carries no goal info; RL²
and VariBAD should learn to explore early and exploit late within an episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GridworldEnv:
    grid_size: int = 5
    episode_length: int = 50

    @property
    def obs_size(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return 4

    def _start_xy(self) -> chex.Array:
        c = self.grid_size // 2
        return jnp.asarray([c, c], dtype=jnp.int32)

    def _obs(self, xy: chex.Array) -> chex.Array:
        return xy.astype(jnp.float32) / jnp.float32(self.grid_size - 1)

    def _sample_goal(self, key: chex.PRNGKey) -> chex.Array:
        # Uniform over the (n_cells - 1) non-center cells via remap trick.
        n = self.grid_size * self.grid_size
        c = self.grid_size // 2
        center_idx = c * self.grid_size + c
        idx = jax.random.randint(key, (), 0, n - 1)
        idx = jnp.where(idx >= center_idx, idx + 1, idx)
        gx = idx // self.grid_size
        gy = idx % self.grid_size
        return jnp.stack([gx, gy]).astype(jnp.int32)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        goal = self._sample_goal(key)
        xy = self._start_xy()
        state = {
            "xy": xy,
            "goal": goal,
            "t": jnp.asarray(0, dtype=jnp.int32),
        }
        return state, self._obs(xy)

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        # Actions: 0=up (+y), 1=down (−y), 2=left (−x), 3=right (+x).
        dx = jnp.asarray([0, 0, -1, 1], dtype=jnp.int32)[action.astype(jnp.int32)]
        dy = jnp.asarray([1, -1, 0, 0], dtype=jnp.int32)[action.astype(jnp.int32)]
        xy = state["xy"]
        new_xy = jnp.stack([
            jnp.clip(xy[0] + dx, 0, self.grid_size - 1),
            jnp.clip(xy[1] + dy, 0, self.grid_size - 1),
        ])
        reward = jnp.all(new_xy == state["goal"]).astype(jnp.float32)

        t_next = state["t"] + 1
        done = t_next >= self.episode_length

        new_goal = jnp.where(done, self._sample_goal(key), state["goal"])
        xy_after = jnp.where(done, self._start_xy(), new_xy)
        t_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)

        new_state = {"xy": xy_after, "goal": new_goal, "t": t_after}
        info = {
            "goal": state["goal"],
            "at_goal": reward,
        }
        return new_state, self._obs(xy_after), reward, done, info
