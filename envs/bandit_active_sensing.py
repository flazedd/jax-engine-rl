"""Contextual Bandit with Active Sensing + Abstain.

Explore-exploit under uncertainty: the action you take affects the quality
of your next observation. This creates a genuine exploration-exploitation
tradeoff where choosing to exploit (pull the best arm) degrades future
information, while exploring (pulling a wrong arm) yields clearer signals.

Action-dependent observation noise:
  - Pull the best arm (exploit) → next side signal σ = σ_exploit (4.0, noisy)
  - Pull a wrong arm (explore) → next side signal σ = σ_explore (1.5, clear)
  - Abstain or first step      → next side signal σ = σ_default (2.5, moderate)

This tests whether agents learn to:
  1. Explore early (sacrifice reward for information)
  2. Abstain when uncertain (safe fallback while gathering info)
  3. Exploit once confident (commit to the best arm)

Actions 0..K-1: pull arm i → Bernoulli(arm_probs[i]) reward
Action K:       abstain    → deterministic ABSTAIN_PAYOFF reward
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .bandit_context import SIDE_MEANS


ABSTAIN_PAYOFF = 0.50

# Action-dependent observation noise levels
SIGMA_EXPLOIT = 4.0    # noisy — pulling correct arm obscures next signal
SIGMA_EXPLORE = 2.5    # moderate — pulling wrong arm reveals next signal
SIGMA_DEFAULT = 2.5    # moderate — abstain or initial observation


class BanditActiveSensingParams(NamedTuple):
    """Immutable parameters for active sensing bandit."""
    n_arms: int = 5
    t_episode: int = 10
    side_dim: int = 4
    sigma_exploit: float = SIGMA_EXPLOIT
    sigma_explore: float = SIGMA_EXPLORE
    sigma_default: float = SIGMA_DEFAULT
    arm_prob_high: float = 0.7
    arm_prob_low: float = 0.35
    abstain_payoff: float = ABSTAIN_PAYOFF


class BanditActiveSensingState(NamedTuple):
    """Mutable bandit state."""
    arm_probs: jnp.ndarray        # (n_arms,)
    task_id: jnp.ndarray          # int32 scalar
    step: jnp.ndarray             # int32 scalar
    prev_action: jnp.ndarray      # int32 scalar — determines next obs noise


def _get_sigma(prev_action, task_id, params):
    """Compute observation noise based on previous action.

    - prev_action == task_id (exploit): σ_exploit (noisy)
    - prev_action < n_arms and != task_id (explore): σ_explore (clear)
    - prev_action == n_arms (abstain): σ_default (moderate)
    """
    is_abstain = (prev_action == params.n_arms)
    is_correct = (prev_action == task_id)
    sigma = jnp.where(
        is_abstain, params.sigma_default,
        jnp.where(is_correct, params.sigma_exploit, params.sigma_explore))
    return sigma


def env_reset(key, params):
    """Sample task, emit initial side-signal obs with default noise."""
    k_task, k_side = jax.random.split(key)
    task_id = jax.random.randint(k_task, (), 0, params.n_arms)
    arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
    arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
    # prev_action = n_arms (abstain equivalent) → σ_default for first obs
    state = BanditActiveSensingState(
        arm_probs=arm_probs, task_id=task_id,
        step=jnp.int32(0), prev_action=jnp.int32(params.n_arms))
    side = SIDE_MEANS[task_id] + params.sigma_default * jax.random.normal(
        k_side, (params.side_dim,))
    obs = jnp.concatenate([jnp.zeros(1), side])
    return state, obs


def env_step(key, state, action, params):
    """Pull arm or abstain. Observation noise depends on action taken."""
    k_rew, k_side = jax.random.split(key)
    is_abstain = (action == params.n_arms)
    safe_action = jnp.clip(action, 0, params.n_arms - 1)
    arm_reward = jax.random.bernoulli(
        k_rew, state.arm_probs[safe_action]).astype(jnp.float32)
    reward = jnp.where(is_abstain, params.abstain_payoff, arm_reward)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = BanditActiveSensingState(
        arm_probs=state.arm_probs, task_id=state.task_id,
        step=new_step, prev_action=action)
    # Next observation noise depends on this action
    sigma = _get_sigma(action, state.task_id, params)
    side = SIDE_MEANS[state.task_id] + sigma * jax.random.normal(
        k_side, (params.side_dim,))
    time_sig = new_step.astype(jnp.float32) / params.t_episode
    obs = jnp.concatenate([time_sig[None], side])
    return new_state, obs, reward, done, jnp.int32(0)
