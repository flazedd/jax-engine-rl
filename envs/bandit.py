"""Bernoulli Bandit — JAX meta-RL validation environment.

K-armed bandit with hidden reward probabilities sampled from Beta(α, β).
Each env_reset samples new arm probs (new task). Observations are minimal:
obs = (step / t_episode,) — just a normalized time signal.

The meta-RL signal comes through prev_action and prev_reward in the
augmented input to recurrent agents. PPO MLP only sees the time signal,
so it can't learn about the current task — establishing a memoryless floor.

Interface matches lob_sim.jax_env:
    env_reset(key, params) -> (state, obs)
    env_step(key, state, action, params) -> (state, obs, reward, done, info)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class BanditParams(NamedTuple):
    """Immutable bandit parameters."""
    n_arms: int = 3
    t_episode: int = 10
    prior_alpha: float = 1.0   # Beta prior on arm probs
    prior_beta: float = 1.0    # Beta(1,1) = Uniform(0,1)


class BanditState(NamedTuple):
    """Mutable bandit state."""
    arm_probs: jnp.ndarray   # (n_arms,) hidden reward probabilities
    step: jnp.ndarray        # int32 scalar


def env_reset(key, params):
    """Sample new arm probs and reset step counter."""
    arm_probs = jax.random.beta(
        key, params.prior_alpha, params.prior_beta, shape=(params.n_arms,))
    state = BanditState(arm_probs=arm_probs, step=jnp.int32(0))
    obs = jnp.zeros(1)  # step 0 → normalized time = 0.0
    return state, obs


def env_step(key, state, action, params):
    """Pull arm `action`, receive Bernoulli(arm_probs[action]) reward."""
    reward = jax.random.bernoulli(key, state.arm_probs[action]).astype(jnp.float32)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = BanditState(arm_probs=state.arm_probs, step=new_step)
    obs = jnp.array([new_step.astype(jnp.float32) / params.t_episode])
    return new_state, obs, reward, done, jnp.int32(0)
