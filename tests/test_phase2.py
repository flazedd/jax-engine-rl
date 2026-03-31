"""Phase 2 tests — sim loop, obs, reward."""
import time

import pytest
import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.state import OrderBookState, init_state
from lob_sim.obs import observe, obs_size
from lob_sim.reward import compute_reward
from lob_sim.step import make_step_fn, run_episode

T = 2000  # steps for integration tests


@pytest.fixture
def config():
    return SimConfig()


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def state(config, key):
    return init_state(config, key)


# ---------- Observation ----------


class TestObservation:
    def test_output_shape(self, state, config):
        """observe() returns array of shape (obs_size(),)."""
        obs = observe(state, config)
        assert obs.shape == (obs_size(),)

    def test_obs_size_matches(self):
        """obs_size() returns 3 * 10 + 3 = 33."""
        assert obs_size() == 33

    def test_no_nan(self, state, config):
        """Observation should have no NaN values for a fresh state."""
        obs = observe(state, config)
        assert not jnp.any(jnp.isnan(obs))

    def test_jit_compatible(self, state, config):
        """jax.jit(observe) works."""
        jit_obs = jax.jit(observe, static_argnums=(1,))(state, config)
        assert jit_obs.shape == (obs_size(),)
        assert not jnp.any(jnp.isnan(jit_obs))


# ---------- Reward ----------


class TestReward:
    def test_zero_when_nothing_changes(self, state, config):
        """If state doesn't change, reward should be ~0 (minus small inventory penalty)."""
        reward = compute_reward(state, state, config)
        # inventory=0 → penalty=0, value unchanged → reward=0
        assert jnp.abs(reward) < 1e-6

    def test_positive_when_profitable(self, state, config):
        """Reward positive when portfolio value increases."""
        next_state = state._replace(cash=jnp.float32(100.0))
        reward = compute_reward(state, next_state, config)
        assert reward > 0

    def test_inventory_penalty(self, state, config):
        """Higher inventory → more negative penalty component."""
        mid = float(state.mid_price)
        # Both next states have the same total value = 500
        next_low = state._replace(
            inventory=jnp.float32(5.0),
            cash=jnp.float32(500.0 - 5.0 * mid),
        )
        next_high = state._replace(
            inventory=jnp.float32(20.0),
            cash=jnp.float32(500.0 - 20.0 * mid),
        )
        r_low = compute_reward(state, next_low, config)
        r_high = compute_reward(state, next_high, config)
        assert r_low > r_high


# ---------- Step Function ----------


class TestStepFunction:
    def test_jit_compiles(self, config, key):
        """jax.jit(step_fn) runs without tracing error."""
        state = init_state(config, key)
        step_fn = make_step_fn(config)
        action = jnp.array([3.0, 3.0])
        jit_step = jax.jit(step_fn)
        new_state, outputs = jit_step(state, action)
        assert new_state.mid_price.shape == ()

    def test_step_count_increments(self, config, key):
        """After one step, step_count should be 1."""
        state = init_state(config, key)
        step_fn = make_step_fn(config)
        action = jnp.array([3.0, 3.0])
        new_state, _ = step_fn(state, action)
        assert int(new_state.step_count) == 1

    def test_agent_orders_removed_after_step(self, key):
        """Unfilled agent volume should not accumulate in the book."""
        # Use a config with no background flow so volume changes are only from agent
        config = SimConfig(
            market_buy_prob=0.0,
            market_sell_prob=0.0,
            cancel_prob=0.0,
            limit_order_rate=0.0,
        )
        state = init_state(config, key)
        step_fn = make_step_fn(config)
        action = jnp.array([3.0, 3.0])

        init_vol = float(jnp.sum(state.bid_volumes) + jnp.sum(state.ask_volumes))

        state1, _ = step_fn(state, action)
        state2, _ = step_fn(state1, action)
        vol2 = float(jnp.sum(state2.bid_volumes) + jnp.sum(state2.ask_volumes))

        # With no background flow and no fills, volume should be unchanged
        assert abs(vol2 - init_vol) < 1.0

    def test_done_on_max_inventory(self, config, key):
        """Done flag triggers when abs(inventory) >= max_inventory."""
        state = init_state(config, key)
        # Set inventory just at the limit
        state = state._replace(inventory=jnp.float32(float(config.max_inventory)))
        step_fn = make_step_fn(config)
        action = jnp.array([3.0, 3.0])
        new_state, _ = step_fn(state, action)
        assert bool(new_state.done)


# ---------- Run Episode ----------


class TestRunEpisode:
    def test_lax_scan_compiles(self, config, key):
        """run_episode with T steps compiles under jax.jit."""
        actions = jnp.broadcast_to(jnp.array([3.0, 3.0]), (T, 2))
        jit_run = jax.jit(run_episode, static_argnums=(0,))
        final, outputs = jit_run(config, key, actions)
        assert final.step_count == T

    def test_mid_price_not_flat(self, config, key):
        """Mid-price trajectory must NOT be constant — this is the #1 sanity check."""
        actions = jnp.broadcast_to(jnp.array([0.0, 0.0]), (T, 2))
        jit_run = jax.jit(run_episode, static_argnums=(0,))
        _, outputs = jit_run(config, key, actions)
        assert float(jnp.std(outputs['mid_price'])) > 0.001

    def test_fills_happen(self, config, key):
        """At least some bid or ask fills should occur over T steps."""
        actions = jnp.broadcast_to(jnp.array([0.0, 0.0]), (T, 2))
        jit_run = jax.jit(run_episode, static_argnums=(0,))
        _, outputs = jit_run(config, key, actions)
        total_fills = float(jnp.sum(outputs['bid_fill']) + jnp.sum(outputs['ask_fill']))
        assert total_fills > 0

    def test_inventory_fluctuates(self, config, key):
        """Inventory should change over time (not stuck at 0)."""
        actions = jnp.broadcast_to(jnp.array([0.0, 0.0]), (T, 2))
        jit_run = jax.jit(run_episode, static_argnums=(0,))
        _, outputs = jit_run(config, key, actions)
        assert float(jnp.max(jnp.abs(outputs['inventory']))) > 0

    def test_output_shapes(self, config, key):
        """All output arrays have leading dim T."""
        actions = jnp.broadcast_to(jnp.array([0.0, 0.0]), (T, 2))
        jit_run = jax.jit(run_episode, static_argnums=(0,))
        _, outputs = jit_run(config, key, actions)

        assert outputs['obs'].shape == (T, obs_size())
        assert outputs['reward'].shape == (T,)
        assert outputs['mid_price'].shape == (T,)
        assert outputs['spread'].shape == (T,)
        assert outputs['inventory'].shape == (T,)
        assert outputs['cash'].shape == (T,)
        assert outputs['bid_fill'].shape == (T,)
        assert outputs['ask_fill'].shape == (T,)
        assert outputs['done'].shape == (T,)
        assert outputs['bid_volumes'].shape == (T, 20)
        assert outputs['ask_volumes'].shape == (T, 20)

    def test_throughput(self, config, key):
        """Warm throughput > 5,000 steps/sec single env."""
        actions = jnp.broadcast_to(jnp.array([0.0, 0.0]), (T, 2))
        jit_run = jax.jit(run_episode, static_argnums=(0,))

        # Warm-up (JIT compilation)
        _, outputs = jit_run(config, key, actions)
        outputs['mid_price'].block_until_ready()

        # Timed run
        key2 = jax.random.PRNGKey(123)
        start = time.time()
        _, outputs = jit_run(config, key2, actions)
        outputs['mid_price'].block_until_ready()
        elapsed = time.time() - start

        steps_per_sec = T / elapsed
        assert steps_per_sec > 5000, f"Only {steps_per_sec:.0f} steps/sec (need > 5000)"
