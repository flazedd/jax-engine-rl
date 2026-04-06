"""Tests for the JAX-native market making environment."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lob_sim.jax_env import (
    EnvParams, EnvState, env_reset, env_step, get_obs,
    rollout_episode, batch_rollout,
)


@pytest.fixture
def params():
    return EnvParams.default()


class TestJaxEnvBasics:
    def test_params_default(self, params):
        assert params.n_actions == 3
        assert params.n_regimes == 3
        assert params.max_inv == 10
        assert params.episode_length == 200
        assert params.fill_bid.shape == (3, 3)
        assert params.fill_ask.shape == (3, 3)

    def test_reset_returns_state_and_obs(self, params):
        key = jax.random.PRNGKey(0)
        state, obs = env_reset(key, params)
        assert obs.shape == (3,)
        assert obs.dtype == jnp.float32

    def test_initial_obs_zero_inv_and_fills(self, params):
        key = jax.random.PRNGKey(0)
        state, obs = env_reset(key, params)
        # inventory=0 → obs[0]=0, no fills → obs[1]=obs[2]=0
        assert obs[0] == 0.0
        assert obs[1] == 0.0
        assert obs[2] == 0.0

    def test_reset_regime_valid(self, params):
        key = jax.random.PRNGKey(42)
        state, _ = env_reset(key, params)
        assert 0 <= int(state.regime) < params.n_regimes

    def test_step_returns_correct_shapes(self, params):
        key = jax.random.PRNGKey(0)
        state, obs = env_reset(key, params)
        k_step = jax.random.PRNGKey(1)
        new_state, new_obs, reward, done, info = env_step(k_step, state, jnp.int32(0), params)
        assert new_obs.shape == (3,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_step_count_increments(self, params):
        key = jax.random.PRNGKey(0)
        state, _ = env_reset(key, params)
        assert int(state.step_count) == 0
        new_state, _, _, _, _ = env_step(jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert int(new_state.step_count) == 1

    def test_episode_terminates_at_200(self, params):
        key = jax.random.PRNGKey(0)
        state, _ = env_reset(key, params)
        # Manually set step_count to 199
        state = state._replace(step_count=jnp.int32(199))
        new_state, _, _, done, _ = env_step(jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert bool(done)

    def test_episode_not_done_before_200(self, params):
        key = jax.random.PRNGKey(0)
        state, _ = env_reset(key, params)
        state = state._replace(step_count=jnp.int32(198))
        new_state, _, _, done, _ = env_step(jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert not bool(done)

    def test_inventory_clipped(self, params):
        key = jax.random.PRNGKey(0)
        state, _ = env_reset(key, params)
        state = state._replace(inventory=jnp.int32(params.max_inv))
        new_state, _, _, _, _ = env_step(jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert int(new_state.inventory) <= params.max_inv
        assert int(new_state.inventory) >= -params.max_inv


class TestJaxEnvJit:
    def test_reset_jit(self, params):
        jit_reset = jax.jit(env_reset)
        state, obs = jit_reset(jax.random.PRNGKey(0), params)
        assert obs.shape == (3,)

    def test_step_jit(self, params):
        state, _ = env_reset(jax.random.PRNGKey(0), params)
        jit_step = jax.jit(env_step)
        new_state, obs, reward, done, info = jit_step(
            jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert obs.shape == (3,)

    def test_vmap_reset(self, params):
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        states, obs = jax.vmap(env_reset, in_axes=(0, None))(keys, params)
        assert obs.shape == (8, 3)


class TestJaxEnvDynamics:
    def test_deterministic_with_same_key(self, params):
        key = jax.random.PRNGKey(42)
        state, _ = env_reset(key, params)

        k = jax.random.PRNGKey(99)
        s1, o1, r1, d1, _ = env_step(k, state, jnp.int32(0), params)
        s2, o2, r2, d2, _ = env_step(k, state, jnp.int32(0), params)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))
        np.testing.assert_array_equal(np.array(o1), np.array(o2))

    def test_fills_binary(self, params):
        """Fill indicators should be 0 or 1."""
        key = jax.random.PRNGKey(0)
        state, _ = env_reset(key, params)
        new_state, obs, _, _, _ = env_step(jax.random.PRNGKey(1), state, jnp.int32(0), params)
        assert float(new_state.last_bid) in (0.0, 1.0)
        assert float(new_state.last_ask) in (0.0, 1.0)

    def test_reward_in_expected_range(self):
        """Regime-blind (always a0) mean reward should be ~2.96/step."""
        params = EnvParams.default()

        def blind_policy(key, obs):
            return jnp.int32(0)

        # Average over many episodes
        keys = jax.random.split(jax.random.PRNGKey(0), 100)
        traj = jax.vmap(rollout_episode, in_axes=(0, None, None, None))(
            keys, blind_policy, params, 200)
        mean_reward = float(jnp.mean(traj["rewards"]))
        assert 2.0 < mean_reward < 4.0, \
            f"Blind per-step reward {mean_reward:.2f} outside expected range"


class TestRollout:
    def test_rollout_episode_shapes(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_episode(jax.random.PRNGKey(0), policy, params, 200)
        assert traj["obs"].shape == (200, 3)
        assert traj["actions"].shape == (200,)
        assert traj["rewards"].shape == (200,)
        assert traj["dones"].shape == (200,)
        assert traj["regimes"].shape == (200,)

    def test_batch_rollout_shapes(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = batch_rollout(jax.random.PRNGKey(0), policy, params, n_envs=4, episode_length=50)
        assert traj["obs"].shape == (4, 50, 3)
        assert traj["rewards"].shape == (4, 50)

    def test_rollout_last_step_done(self, params):
        def policy(key, obs):
            return jnp.int32(0)

        traj = rollout_episode(jax.random.PRNGKey(0), policy, params, 200)
        assert bool(traj["dones"][-1])
        assert not bool(traj["dones"][0])
