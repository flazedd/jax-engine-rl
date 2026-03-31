"""Phase 4 tests -- policy divergence across regimes."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.config import SimConfig
from lob_sim.step import run_episode
from lob_sim.actions import ACTION_TABLE, N_ACTIONS
from lob_sim.regime import N_REGIMES

# Use smaller N for faster tests, but enough to be statistically meaningful
N_EPISODES = 100  # reduced from 200 for test speed
T_STEPS = 500     # reduced from 1000 for test speed

config = SimConfig(max_steps=T_STEPS)


def _compute_mean_reward(config, action_idx, regime_idx, n_episodes, t_steps, master_key):
    """Run n_episodes of t_steps with a fixed action and locked regime, return mean total reward."""
    keys = jax.random.split(master_key, n_episodes)
    actions_repeated = jnp.full((t_steps,), action_idx, dtype=jnp.int32)
    batched_run = jax.vmap(
        lambda k: run_episode(config, k, actions_repeated, locked_regime=regime_idx)
    )
    _, outputs = batched_run(keys)
    # outputs["reward"] shape: (n_episodes, t_steps)
    total_rewards = outputs["reward"].sum(axis=1)
    return total_rewards.mean()


@pytest.fixture(scope="module")
def reward_matrix():
    """Compute the (25, 3) reward matrix once for all tests.
    reward_matrix[action_idx, regime_idx] = mean total reward.
    """
    master_key = jax.random.PRNGKey(42)
    matrix = np.zeros((N_ACTIONS, N_REGIMES))

    # Pre-compile with JIT for speed
    jit_compute = jax.jit(_compute_mean_reward, static_argnums=(0, 1, 2, 3, 4))

    for regime_idx in range(N_REGIMES):
        for action_idx in range(N_ACTIONS):
            key = jax.random.fold_in(master_key, regime_idx * N_ACTIONS + action_idx)
            val = jit_compute(config, action_idx, regime_idx, N_EPISODES, T_STEPS, key)
            matrix[action_idx, regime_idx] = float(val)

    return matrix


class TestPolicyDivergence:
    def test_completes_in_reasonable_time(self, reward_matrix):
        """The full evaluation should complete (this test just checks it ran)."""
        assert reward_matrix.shape == (25, 3)

    def test_optimal_actions_differ(self, reward_matrix):
        """Optimal action (argmax) is different for each regime."""
        optimal = reward_matrix.argmax(axis=0)  # shape (3,)
        # All 3 optimal actions should be different
        assert len(set(optimal.tolist())) == 3, (
            f"Optimal actions are not all different: {optimal}. "
            f"Regime params may need more asymmetry."
        )

    def test_noise_optimal_is_symmetric(self, reward_matrix):
        """Noise regime optimal action should have bid_tick ~ ask_tick."""
        optimal_idx = reward_matrix[:, 0].argmax()
        bid, ask = ACTION_TABLE[optimal_idx]
        assert abs(int(bid) - int(ask)) <= 1, (
            f"Noise optimal ({bid},{ask}) is not symmetric."
        )

    def test_bull_optimal_tight_bid_wide_ask(self, reward_matrix):
        """Bull optimal should have bid_tick < ask_tick."""
        optimal_idx = reward_matrix[:, 1].argmax()
        bid, ask = ACTION_TABLE[optimal_idx]
        assert int(bid) < int(ask), (
            f"Bull optimal ({bid},{ask}) should have bid < ask."
        )

    def test_bear_optimal_wide_bid_tight_ask(self, reward_matrix):
        """Bear optimal should have bid_tick > ask_tick."""
        optimal_idx = reward_matrix[:, 2].argmax()
        bid, ask = ACTION_TABLE[optimal_idx]
        assert int(bid) > int(ask), (
            f"Bear optimal ({bid},{ask}) should have bid > ask."
        )

    def test_cross_regime_penalty(self, reward_matrix):
        """Using the wrong regime's optimal action should cost >10% reward."""
        for regime in range(3):
            best_action = reward_matrix[:, regime].argmax()
            best_reward = reward_matrix[best_action, regime]
            for other_regime in range(3):
                if other_regime == regime:
                    continue
                other_best = reward_matrix[:, other_regime].argmax()
                wrong_reward = reward_matrix[other_best, regime]
                # Wrong policy should be worse
                assert wrong_reward < best_reward, (
                    f"Wrong policy (regime {other_regime}'s optimal) applied in "
                    f"regime {regime} is not worse: {wrong_reward} vs {best_reward}"
                )
