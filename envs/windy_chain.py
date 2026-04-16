"""Windy Chain MDP — regime-dependent stochastic transitions + side signals.

N-state chain with hidden regime that determines a goal position, a
stochastic wind pattern, AND noisy side-signal observations.  Designed
to differentiate VariBAD(+HN) from RL²(+HN):

  1. **Transitions** are regime-dependent (wind) → ELBO learns
     P(next_pos | pos, action, z) with supervised gradients every step.
  2. **Side signals** are regime-dependent Gaussians → ELBO also learns
     P(side | z), providing additional dense gradient on regime identity.
  3. **Reward** is noisy Bernoulli at goal → RL signal is sparse and
     stochastic, slowing reward-only regime identification.

RL²(+HN) must discover all of this purely through RL credit assignment.
VariBAD(+HN) gets direct supervised loss on transitions AND observations,
giving it a significant sample-efficiency advantage.

States:  0, 1, ..., N-1 arranged in a chain (N=7).
Actions: 0=left, 1=stay, 2=right.
Regimes: 3 hidden regimes.

  Regime 0: goal=0, strong rightward wind, side mean [+1, +1, 0, 0]
  Regime 1: goal=3, calm,                  side mean [-1, +1, +1, 0]
  Regime 2: goal=6, strong leftward wind,  side mean [ 0, -1, -1, +1]

Transition: next_pos = clip(pos + intended + wind, 0, N-1)
            wind ~ Categorical({-1, 0, +1}, WIND_PROBS[regime])
Reward:     Bernoulli(reward_prob) if at goal, else 0.
Observation: [one_hot(pos, N), side_0, side_1, side_2, side_3] — (N+4)D

Interface matches envs/chain.py:
    env_reset(key, params) -> (state, obs)
    env_step(key, state, action, params) -> (state, obs, reward, done, info)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Regime definitions (module-level constants)
# ---------------------------------------------------------------------------

# Goal position per regime
GOALS = jnp.array([0, 3, 6], dtype=jnp.int32)

# Wind probabilities: [p_left_wind, p_no_wind, p_right_wind] per regime.
# Stronger wind (0.5) → transitions are very informative about regime.
WIND_PROBS = jnp.array([
    [0.0, 0.5, 0.5],   # regime 0: strong rightward wind (opposes goal at 0)
    [0.0, 1.0, 0.0],   # regime 1: calm (goal at centre 3)
    [0.5, 0.5, 0.0],   # regime 2: strong leftward wind (opposes goal at 6)
])

# Side-signal cluster centres — 4D, well-separated across regimes.
# Min pairwise L2 distance ≈ 2.45. At σ=1.5 each single obs is noisy,
# but ELBO can learn the mean and accumulate evidence over steps.
SIDE_MEANS = jnp.array([
    [+1.0, +1.0,  0.0,  0.0],   # regime 0
    [-1.0, +1.0, +1.0,  0.0],   # regime 1
    [ 0.0, -1.0, -1.0, +1.0],   # regime 2
])

SIDE_DIM = 4
SIDE_SIGMA = 1.5


# ---------------------------------------------------------------------------
# State / params
# ---------------------------------------------------------------------------

class WindyChainParams(NamedTuple):
    """Immutable windy chain MDP parameters."""
    n_states: int = 7
    n_regimes: int = 3
    t_episode: int = 30
    n_actions: int = 3       # left=0, stay=1, right=2
    reward_prob: float = 0.4  # Bernoulli prob at goal (noisy)
    side_dim: int = SIDE_DIM
    side_sigma: float = SIDE_SIGMA


class WindyChainState(NamedTuple):
    """Mutable windy chain MDP state."""
    pos: jnp.ndarray       # int32 scalar — current position
    regime: jnp.ndarray    # int32 scalar — hidden regime
    step: jnp.ndarray      # int32 scalar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(pos, regime, n_states, key, side_sigma):
    """Build observation: [one_hot(pos), side_signals]."""
    pos_oh = jax.nn.one_hot(pos, n_states)
    side = SIDE_MEANS[regime] + side_sigma * jax.random.normal(key, (SIDE_DIM,))
    return jnp.concatenate([pos_oh, side])


# ---------------------------------------------------------------------------
# Core env functions
# ---------------------------------------------------------------------------

def env_reset(key, params):
    """Sample regime and start at random position."""
    k_pos, k_regime, k_side = jax.random.split(key, 3)
    pos = jax.random.randint(k_pos, (), 0, params.n_states)
    regime = jax.random.randint(k_regime, (), 0, params.n_regimes)
    state = WindyChainState(
        pos=jnp.int32(pos), regime=jnp.int32(regime), step=jnp.int32(0))
    obs = _make_obs(pos, regime, params.n_states, k_side, params.side_sigma)
    return state, obs


def env_step(key, state, action, params):
    """Take action with regime-dependent wind, receive noisy reward.

    Returns (new_state, obs, reward, done, info) where info = regime.
    """
    k_wind, k_rew, k_side = jax.random.split(key, 3)

    # Intended movement: left=-1, stay=0, right=+1
    intended = jnp.int32(action) - 1

    # Sample wind from regime-specific distribution
    wind_logits = jnp.log(jnp.clip(WIND_PROBS[state.regime], 1e-30, None))
    wind_idx = jax.random.categorical(k_wind, wind_logits)
    wind = jnp.int32(wind_idx) - 1   # {-1, 0, +1}

    # Apply movement + wind, clip to bounds
    new_pos = jnp.clip(state.pos + intended + wind, 0, params.n_states - 1)
    new_pos = jnp.int32(new_pos)

    # Noisy Bernoulli reward at goal
    at_goal = (new_pos == GOALS[state.regime])
    reward = jnp.where(
        at_goal,
        jax.random.bernoulli(k_rew, params.reward_prob).astype(jnp.float32),
        0.0)

    new_step = state.step + 1
    done = (new_step >= params.t_episode)

    new_state = WindyChainState(
        pos=new_pos, regime=state.regime, step=new_step)
    obs = _make_obs(new_pos, state.regime, params.n_states, k_side,
                    params.side_sigma)
    return new_state, obs, reward, done, state.regime
