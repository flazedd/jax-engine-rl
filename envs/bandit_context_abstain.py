"""Contextual Bandit with Side Information + Abstain action.

Calibration-stress variant: adds a fixed-payoff "abstain" action (action K).
The optimal policy abstains while posterior uncertainty is high and commits
to the MAP arm once confidence exceeds a threshold. Methods with
miscalibrated posteriors abstain too early (overconfident) or too late
(underconfident) — a behaviour measurable via the abstain-rate curve.

Actions 0..K-1: pull arm i → Bernoulli(arm_probs[i]) reward
Action K:       abstain    → deterministic ABSTAIN_PAYOFF reward

With arm_prob_high=0.7, arm_prob_low=0.35, K=5:
  E[random pull] = (0.7 + 4*0.35)/5 = 0.42
  E[optimal pull] = 0.7
  ABSTAIN_PAYOFF = 0.50  (above random, below optimal)

Break-even: pull MAP arm when MAP belief > (0.50 - 0.35)/(0.70 - 0.35) ≈ 0.43
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .bandit_context import SIDE_MEANS


ABSTAIN_PAYOFF = 0.50


class BanditAbstainParams(NamedTuple):
    """Immutable parameters for abstain bandit."""
    n_arms: int = 5
    t_episode: int = 15
    side_dim: int = 4
    side_sigma: float = 2.5
    arm_prob_high: float = 0.7
    arm_prob_low: float = 0.35
    abstain_payoff: float = ABSTAIN_PAYOFF


class BanditAbstainState(NamedTuple):
    """Mutable bandit state."""
    arm_probs: jnp.ndarray        # (n_arms,)
    task_id: jnp.ndarray          # int32 scalar
    step: jnp.ndarray             # int32 scalar


def env_reset(key, params):
    """Sample task, emit initial side-signal obs."""
    k_task, k_side = jax.random.split(key)
    task_id = jax.random.randint(k_task, (), 0, params.n_arms)
    arm_probs = jnp.full(params.n_arms, params.arm_prob_low)
    arm_probs = arm_probs.at[task_id].set(params.arm_prob_high)
    state = BanditAbstainState(arm_probs=arm_probs, task_id=task_id,
                               step=jnp.int32(0))
    side = SIDE_MEANS[task_id] + params.side_sigma * jax.random.normal(
        k_side, (params.side_dim,))
    obs = jnp.concatenate([jnp.zeros(1), side])
    return state, obs


def env_step(key, state, action, params):
    """Pull arm or abstain. Abstain = action n_arms."""
    k_rew, k_side = jax.random.split(key)
    is_abstain = (action == params.n_arms)
    arm_reward = jax.random.bernoulli(
        k_rew, state.arm_probs[jnp.clip(action, 0, params.n_arms - 1)]
    ).astype(jnp.float32)
    reward = jnp.where(is_abstain, params.abstain_payoff, arm_reward)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = BanditAbstainState(
        arm_probs=state.arm_probs, task_id=state.task_id, step=new_step)
    side = SIDE_MEANS[state.task_id] + params.side_sigma * jax.random.normal(
        k_side, (params.side_dim,))
    time_sig = new_step.astype(jnp.float32) / params.t_episode
    obs = jnp.concatenate([time_sig[None], side])
    return new_state, obs, reward, done, jnp.int32(0)
