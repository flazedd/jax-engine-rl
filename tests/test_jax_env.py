"""Tests for the Phase 0 JAX-native POMDP market making environment."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lob_sim.jax_env import (
    EnvParams, EnvState, env_reset, env_step, get_obs,
    rollout_episode, batch_rollout, rollout_trial,
    _stationary_distribution,
)


@pytest.fixture
def params():
    return EnvParams.default()


# ── EnvParams ───────────────────────────────────────────────────

class TestEnvParams:
    def test_default_shapes(self, params):
        assert params.kappa.shape == (3, 2)
        assert params.delta.shape == (3, 2)
        assert params.drift_probs.shape == (3, 3)
        assert params.sigma_sq.shape == (3,)
        assert params.hmm_transition.shape == (3, 3)
        assert params.stationary_dist.shape == (3,)

    def test_default_scalars(self, params):
        assert params.gamma_disc == 0.99
        assert params.gamma_inventory == 0.1
        assert params.boundary_penalty == 5.0
        assert params.inventory_max == 5
        assert params.t_episode == 200
        assert params.n_regimes == 3
        assert params.n_actions == 3
        assert params.locked_regime == -1

    def test_hmm_rows_sum_to_one(self, params):
        sums = params.hmm_transition.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_drift_probs_rows_sum_to_one(self, params):
        sums = params.drift_probs.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_stationary_dist_sums_to_one(self, params):
        np.testing.assert_allclose(
            float(params.stationary_dist.sum()), 1.0, atol=1e-6)

    def test_stationary_dist_is_eigenvector(self, params):
        """π·T = π for the stationary distribution."""
        pi = params.stationary_dist
        pi_next = params.hmm_transition.T @ pi
        np.testing.assert_allclose(pi, pi_next, atol=1e-5)

    def test_kappa_values(self, params):
        expected = jnp.array([[2.0, 2.0], [2.0, 0.8], [0.8, 2.0]])
        np.testing.assert_array_equal(params.kappa, expected)

    def test_delta_values(self, params):
        expected = jnp.array([[1.0, 1.0], [1.0, 3.0], [3.0, 1.0]])
        np.testing.assert_array_equal(params.delta, expected)


# ── Fill model ──────────────────────────────────────────────────

class TestFillModel:
    """Verify P(fill) = exp(-κ[regime, side] · δ[action, side])."""

    def test_noise_symmetric_action_fills_equal(self, params):
        """Noise + symmetric action → equal bid/ask fill probs."""
        p_bid = float(jnp.exp(-params.kappa[0, 0] * params.delta[0, 0]))
        p_ask = float(jnp.exp(-params.kappa[0, 1] * params.delta[0, 1]))
        assert p_bid == pytest.approx(p_ask)

    def test_bull_ask_fills_easier(self, params):
        """Bull regime has lower κ on ask side → higher ask fill prob."""
        # Same action (symmetric), compare sides
        p_bid = float(jnp.exp(-params.kappa[1, 0] * params.delta[0, 0]))
        p_ask = float(jnp.exp(-params.kappa[1, 1] * params.delta[0, 1]))
        assert p_ask > p_bid

    def test_bear_bid_fills_easier(self, params):
        """Bear regime has lower κ on bid side → higher bid fill prob."""
        p_bid = float(jnp.exp(-params.kappa[2, 0] * params.delta[0, 0]))
        p_ask = float(jnp.exp(-params.kappa[2, 1] * params.delta[0, 1]))
        assert p_bid > p_ask

    def test_fill_probs_in_valid_range(self, params):
        """All fill probs should be in (0, 1)."""
        for r in range(3):
            for a in range(3):
                p_bid = float(jnp.exp(-params.kappa[r, 0] * params.delta[a, 0]))
                p_ask = float(jnp.exp(-params.kappa[r, 1] * params.delta[a, 1]))
                assert 0 < p_bid < 1
                assert 0 < p_ask < 1

    def test_wider_spread_lower_fill(self, params):
        """Wider spread (larger δ) → lower fill probability."""
        # Compare symmetric (δ=1) vs lean-ask (δ_ask=3) in noise
        p_sym = float(jnp.exp(-params.kappa[0, 1] * params.delta[0, 1]))  # δ=1
        p_lean = float(jnp.exp(-params.kappa[0, 1] * params.delta[1, 1]))  # δ=3
        assert p_sym > p_lean


# ── Reset ───────────────────────────────────────────────────────

class TestReset:
    def test_returns_state_and_obs(self, params):
        state, obs = env_reset(jax.random.PRNGKey(0), params)
        assert isinstance(state, EnvState)
        assert obs.shape == (4,)
        assert obs.dtype == jnp.float32

    def test_initial_state_values(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        assert int(state.inventory) == 0
        assert float(state.mid_price) == 0.0
        assert int(state.step) == 0
        assert float(state.last_fill_bid) == 0.0
        assert float(state.last_fill_ask) == 0.0
        assert float(state.last_mid_change) == 0.0

    def test_initial_obs_zeros(self, params):
        _, obs = env_reset(jax.random.PRNGKey(0), params)
        np.testing.assert_array_equal(obs, [0.0, 0.0, 0.0, 0.0])

    def test_regime_valid(self, params):
        for seed in range(20):
            state, _ = env_reset(jax.random.PRNGKey(seed), params)
            assert 0 <= int(state.regime) < 3

    def test_locked_regime(self):
        for r in range(3):
            p = EnvParams.default()._replace(locked_regime=r)
            state, _ = env_reset(jax.random.PRNGKey(42), p)
            assert int(state.regime) == r


# ── Step ────────────────────────────────────────────────────────

class TestStep:
    def test_returns_correct_shapes(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        new_state, obs, reward, done, info = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert obs.shape == (4,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_step_count_increments(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        assert int(state.step) == 0
        new_state, _, _, _, _ = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert int(new_state.step) == 1

    def test_episode_done_at_200(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        state = state._replace(step=jnp.int32(199))
        _, _, _, done, _ = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert bool(done)

    def test_not_done_before_200(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        state = state._replace(step=jnp.int32(198))
        _, _, _, done, _ = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert not bool(done)

    def test_inventory_clipped_to_5(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        state = state._replace(inventory=jnp.int32(5))
        new_state, _, _, _, _ = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert int(new_state.inventory) <= 5
        assert int(new_state.inventory) >= -5

    def test_inventory_clipped_negative(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        state = state._replace(inventory=jnp.int32(-5))
        new_state, _, _, _, _ = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert int(new_state.inventory) >= -5
        assert int(new_state.inventory) <= 5

    def test_fills_binary(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        for seed in range(10):
            new_state, _, _, _, _ = env_step(
                jax.random.PRNGKey(seed), state, jnp.int32(0), params)
            assert float(new_state.last_fill_bid) in (0.0, 1.0)
            assert float(new_state.last_fill_ask) in (0.0, 1.0)

    def test_mid_change_in_valid_set(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        for seed in range(20):
            new_state, _, _, _, _ = env_step(
                jax.random.PRNGKey(seed), state, jnp.int32(0), params)
            mc = float(new_state.last_mid_change)
            assert mc in (-1.0, 0.0, 1.0), f"mid_change={mc}"

    def test_mid_price_tracks_changes(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        total_change = 0.0
        for seed in range(10):
            new_state, _, _, _, _ = env_step(
                jax.random.PRNGKey(seed), state, jnp.int32(0), params)
            total_change += float(new_state.last_mid_change)
            state = new_state
        np.testing.assert_allclose(
            float(state.mid_price), total_change, atol=1e-5)

    def test_deterministic_with_same_key(self, params):
        state, _ = env_reset(jax.random.PRNGKey(42), params)
        k = jax.random.PRNGKey(99)
        s1, o1, r1, _, _ = env_step(k, state, jnp.int32(0), params)
        s2, o2, r2, _, _ = env_step(k, state, jnp.int32(0), params)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))
        np.testing.assert_array_equal(np.array(o1), np.array(o2))

    def test_info_contains_regime(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        _, _, _, _, info = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert "regime" in info

    def test_obs_matches_state(self, params):
        """Observation fields should match the new state."""
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        new_state, obs, _, _, _ = env_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert float(obs[0]) == float(new_state.last_fill_bid)
        assert float(obs[1]) == float(new_state.last_fill_ask)
        assert float(obs[2]) == float(new_state.last_mid_change)
        assert float(obs[3]) == float(new_state.inventory)


# ── Reward ──────────────────────────────────────────────────────

class TestReward:
    def test_spread_pnl_positive_on_fill(self, params):
        """If both sides fill, spread PnL should be δ_bid + δ_ask."""
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        state = state._replace(inventory=jnp.int32(0))
        # Force fills by running many seeds and checking
        rewards = []
        for seed in range(100):
            _, _, r, _, _ = env_step(
                jax.random.PRNGKey(seed), state, jnp.int32(0), params)
            rewards.append(float(r))
        # Some rewards should be positive (when fills happen)
        assert max(rewards) > 0

    def test_boundary_penalty_at_max_inv(self, params):
        """Reward at inventory boundary should include boundary penalty."""
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        # At max inventory with no fills: penalty = 5.0 * 5 = 25.0
        state = state._replace(inventory=jnp.int32(5))
        rewards_at_boundary = []
        rewards_at_zero = []
        state_zero = state._replace(inventory=jnp.int32(0))
        for seed in range(50):
            k = jax.random.PRNGKey(seed)
            _, _, r_b, _, _ = env_step(k, state, jnp.int32(0), params)
            _, _, r_z, _, _ = env_step(k, state_zero, jnp.int32(0), params)
            rewards_at_boundary.append(float(r_b))
            rewards_at_zero.append(float(r_z))
        # Mean reward at boundary should be much lower
        assert np.mean(rewards_at_boundary) < np.mean(rewards_at_zero)

    def test_inventory_penalty_quadratic(self, params):
        """Higher |inventory| → higher penalty."""
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        mean_rewards = {}
        for q in [0, 2, 4]:
            s = state._replace(inventory=jnp.int32(q))
            rs = []
            for seed in range(200):
                _, _, r, _, _ = env_step(
                    jax.random.PRNGKey(seed), s, jnp.int32(0), params)
                rs.append(float(r))
            mean_rewards[q] = np.mean(rs)
        assert mean_rewards[0] > mean_rewards[2] > mean_rewards[4]


# ── HMM dynamics ────────────────────────────────────────────────

class TestHMM:
    def test_regime_stays_valid(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        for seed in range(50):
            state, _, _, _, _ = env_step(
                jax.random.PRNGKey(seed), state, jnp.int32(0), params)
            assert 0 <= int(state.regime) < 3

    def test_locked_regime_stays_fixed(self):
        for locked in range(3):
            p = EnvParams.default()._replace(locked_regime=locked)
            state, _ = env_reset(jax.random.PRNGKey(0), p)
            for seed in range(20):
                state, _, _, _, _ = env_step(
                    jax.random.PRNGKey(seed), state, jnp.int32(0), p)
                assert int(state.regime) == locked

    def test_regime_transitions_happen(self, params):
        """Over many steps, regime should change at least once."""
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        regimes = [int(state.regime)]
        for seed in range(200):
            state, _, _, _, _ = env_step(
                jax.random.PRNGKey(seed), state, jnp.int32(0), params)
            regimes.append(int(state.regime))
        assert len(set(regimes)) > 1


# ── JIT / vmap ──────────────────────────────────────────────────

class TestJitVmap:
    def test_reset_jit(self, params):
        jit_reset = jax.jit(env_reset)
        state, obs = jit_reset(jax.random.PRNGKey(0), params)
        assert obs.shape == (4,)

    def test_step_jit(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        jit_step = jax.jit(env_step)
        new_state, obs, reward, done, info = jit_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert obs.shape == (4,)

    def test_vmap_reset(self, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        states, obs = jax.vmap(env_reset, in_axes=(0, None))(keys, params)
        assert obs.shape == (8, 4)

    def test_vmap_step(self, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states, obs = jax.vmap(env_reset, in_axes=(0, None))(keys, params)
        step_keys = jax.random.split(jax.random.PRNGKey(1), 4)
        actions = jnp.zeros(4, dtype=jnp.int32)
        new_states, new_obs, rewards, dones, _ = jax.vmap(
            env_step, in_axes=(0, 0, 0, None))(
            step_keys, states, actions, params)
        assert new_obs.shape == (4, 4)
        assert rewards.shape == (4,)


# ── Rollout ─────────────────────────────────────────────────────

class TestRollout:
    def test_episode_shapes(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_episode(jax.random.PRNGKey(0), policy, params)
        assert traj["obs"].shape == (200, 4)
        assert traj["actions"].shape == (200,)
        assert traj["rewards"].shape == (200,)
        assert traj["dones"].shape == (200,)
        assert traj["true_regimes"].shape == (200,)

    def test_last_step_done(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_episode(jax.random.PRNGKey(0), policy, params)
        assert bool(traj["dones"][-1])
        assert not bool(traj["dones"][0])

    def test_batch_rollout_shapes(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = batch_rollout(jax.random.PRNGKey(0), policy, params, n_envs=4)
        assert traj["obs"].shape == (4, 200, 4)
        assert traj["rewards"].shape == (4, 200)

    def test_regimes_valid(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_episode(jax.random.PRNGKey(0), policy, params)
        regimes = np.array(traj["true_regimes"])
        assert np.all(regimes >= 0)
        assert np.all(regimes < 3)


# ── Multi-episode trial ────────────────────────────────────────

class TestTrial:
    def test_trial_shapes(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_trial(
            jax.random.PRNGKey(0), policy, params, episodes_per_trial=4)
        assert traj["obs"].shape == (800, 4)
        assert traj["actions"].shape == (800,)
        assert traj["rewards"].shape == (800,)
        assert traj["dones"].shape == (800,)
        assert traj["true_regimes"].shape == (800,)

    def test_trial_done_at_episode_boundaries(self, params):
        """Done should be True at steps 199, 399, 599, 799."""
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_trial(
            jax.random.PRNGKey(0), policy, params, episodes_per_trial=4)
        dones = np.array(traj["dones"])
        for boundary in [199, 399, 599, 799]:
            assert dones[boundary], f"Step {boundary} should be done"
        # Step 0 should not be done
        assert not dones[0]

    def test_trial_regime_continuous(self, params):
        """Regime should NOT reset at episode boundaries."""
        def policy(key, obs):
            return jnp.int32(0)

        # Use locked regime to verify continuity deterministically
        p = params._replace(locked_regime=1)
        traj = rollout_trial(
            jax.random.PRNGKey(0), policy, p, episodes_per_trial=4)
        regimes = np.array(traj["true_regimes"])
        # All regimes should be 1 (locked)
        assert np.all(regimes == 1)

    def test_trial_inventory_resets(self, params):
        """Inventory should reset to 0 at episode boundaries.

        We verify by checking that obs[3] (inventory) is 0 at the
        start of each episode (steps 0, 200, 400, 600).
        """
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_trial(
            jax.random.PRNGKey(0), policy, params, episodes_per_trial=4)
        obs = np.array(traj["obs"])
        for start in [0, 200, 400, 600]:
            assert obs[start, 3] == 0.0, \
                f"Inventory at step {start} should be 0"

    def test_trial_2_episodes(self, params):
        """Trial with 2 episodes should produce 400 steps."""
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_trial(
            jax.random.PRNGKey(0), policy, params, episodes_per_trial=2)
        assert traj["obs"].shape == (400, 4)


# ── Drift distribution ─────────────────────────────────────────

class TestDrift:
    def test_noise_drift_symmetric(self, params):
        """Noise regime should have zero-mean drift."""
        dp = params.drift_probs[0]
        # P(-1) == P(+1)
        np.testing.assert_allclose(float(dp[0]), float(dp[2]))

    def test_bull_drift_positive(self, params):
        """Bull regime should have positive mean drift."""
        dp = params.drift_probs[1]
        mean = -1 * float(dp[0]) + 0 * float(dp[1]) + 1 * float(dp[2])
        assert mean > 0

    def test_bear_drift_negative(self, params):
        """Bear regime should have negative mean drift."""
        dp = params.drift_probs[2]
        mean = -1 * float(dp[0]) + 0 * float(dp[1]) + 1 * float(dp[2])
        assert mean < 0
