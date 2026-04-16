"""Discrete-Task Bandit with Abstain — HMM-style regime identification.

4 discrete tasks with qualitatively different reward profiles, mirroring
the market-making regime structure. Two task types:

  Spike tasks (0,1): one dominant arm at 0.9, others at 0.1.
      High upside if identified, high cost if wrong. Worth abstaining longer.
  Split tasks (2,3): two decent arms at 0.55, two weak at 0.25.
      More forgiving — partial identification still helps.

This tests whether agents develop calibrated commitment strategies that
adapt to the task's risk profile, not just whether they can identify tasks.

Actions 0..3: pull arm i → Bernoulli(REWARD_PROFILES[task, i])
Action 4:     abstain    → deterministic ABSTAIN_PAYOFF

With uniform prior over tasks:
  E[random arm pull] = 0.35
  E[best arm]        = 0.45  (= ABSTAIN_PAYOFF, break-even)
  → agents must accumulate evidence before committing
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


N_TASKS = 4
ABSTAIN_PAYOFF = 0.45

# Two "spike" tasks (one dominant arm) + two "split" tasks (two decent arms)
REWARD_PROFILES = jnp.array([
    [0.9, 0.1, 0.1, 0.1],     # task 0: spike at arm 0
    [0.1, 0.1, 0.1, 0.9],     # task 1: spike at arm 3
    [0.55, 0.55, 0.25, 0.25],  # task 2: split arms 0,1
    [0.25, 0.25, 0.55, 0.55],  # task 3: split arms 2,3
])

# Side-signal cluster centres in 4D (one per task)
# Pairwise L2 distances ≈ 2.83; with σ=2.0 need multiple obs
SIDE_MEANS = jnp.array([
    [+1.0, +1.0, +1.0,  0.0],
    [+1.0, -1.0, -1.0,  0.0],
    [-1.0, +1.0, -1.0,  0.0],
    [-1.0, -1.0, +1.0,  0.0],
])


class BanditDiscreteAbstainParams(NamedTuple):
    """Immutable parameters."""
    n_tasks: int = N_TASKS
    n_arms: int = 4
    t_episode: int = 15
    side_dim: int = 4
    side_sigma: float = 2.0
    abstain_payoff: float = ABSTAIN_PAYOFF


class BanditDiscreteAbstainState(NamedTuple):
    """Mutable state."""
    reward_profile: jnp.ndarray   # (n_arms,) — reward probs for this episode
    task_id: jnp.ndarray          # int32 scalar
    step: jnp.ndarray             # int32 scalar


def env_reset(key, params):
    """Sample a discrete task, emit initial side-signal obs."""
    k_task, k_side = jax.random.split(key)
    task_id = jax.random.randint(k_task, (), 0, params.n_tasks)
    reward_profile = REWARD_PROFILES[task_id]
    state = BanditDiscreteAbstainState(
        reward_profile=reward_profile, task_id=task_id,
        step=jnp.int32(0))
    side = SIDE_MEANS[task_id] + params.side_sigma * jax.random.normal(
        k_side, (params.side_dim,))
    obs = jnp.concatenate([jnp.zeros(1), side])
    return state, obs


def env_step(key, state, action, params):
    """Pull arm or abstain. Abstain = action n_arms."""
    k_rew, k_side = jax.random.split(key)
    is_abstain = (action == params.n_arms)
    # Clip action for indexing; reward is overridden if abstaining
    safe_action = jnp.clip(action, 0, params.n_arms - 1)
    arm_reward = jax.random.bernoulli(
        k_rew, state.reward_profile[safe_action]).astype(jnp.float32)
    reward = jnp.where(is_abstain, params.abstain_payoff, arm_reward)
    new_step = state.step + 1
    done = (new_step >= params.t_episode)
    new_state = BanditDiscreteAbstainState(
        reward_profile=state.reward_profile, task_id=state.task_id,
        step=new_step)
    side = SIDE_MEANS[state.task_id] + params.side_sigma * jax.random.normal(
        k_side, (params.side_dim,))
    time_sig = new_step.astype(jnp.float32) / params.t_episode
    obs = jnp.concatenate([time_sig[None], side])
    return new_state, obs, reward, done, jnp.int32(0)
