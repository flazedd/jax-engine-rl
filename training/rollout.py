"""lax.scan-based rollout.

`parallel_envs` copies of an env (via vmap) run for `rollout_length` steps
(via scan). The compiled rollout is called once per training iteration; its
signature is shape-stable so the whole per-iteration (rollout + GAE + update)
step can be a single JIT.

Returns:
  trajectory: dict of stacked per-step arrays with shape [T, parallel_envs, ...].
              Keys: `obs`, `action`, `reward`, `done`, plus every key returned
              in `extras` from the agent's act().
  final_obs:  observation array after the last step, per env. Shape
              [parallel_envs, obs_size]. Used as the bootstrap point for GAE.
"""
from __future__ import annotations

from functools import partial

import chex
import jax


def _vmapped_reset(env, keys):
    return jax.vmap(env.reset)(keys)


def _vmapped_step(env, states, actions, keys):
    return jax.vmap(env.step)(states, actions, keys)


def _vmapped_act(agent, agent_state, obs, keys):
    def act_one(obs_i, key_i):
        return agent.act(agent_state, obs_i, key_i)
    return jax.vmap(act_one)(obs, keys)


@partial(jax.jit, static_argnames=("env", "agent", "parallel_envs", "rollout_length"))
def rollout(
    env,
    agent,
    agent_state: chex.ArrayTree,
    key: chex.PRNGKey,
    parallel_envs: int,
    rollout_length: int,
) -> tuple[dict[str, chex.Array], chex.Array]:
    reset_key, key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, parallel_envs)
    env_states, obs = _vmapped_reset(env, reset_keys)

    def scan_step(carry, _):
        env_states, obs, key = carry
        act_key, step_key, key = jax.random.split(key, 3)
        act_keys = jax.random.split(act_key, parallel_envs)
        actions, extras, _ = _vmapped_act(agent, agent_state, obs, act_keys)
        step_keys = jax.random.split(step_key, parallel_envs)
        new_states, next_obs, rewards, dones, _info = _vmapped_step(
            env, env_states, actions, step_keys
        )
        out = {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "done": dones,
        }
        out.update({k: v for k, v in extras.items()})
        return (new_states, next_obs, key), out

    (final_states, final_obs, _), trajectory = jax.lax.scan(
        scan_step, (env_states, obs, key), xs=None, length=rollout_length
    )
    return trajectory, final_obs
