"""Phase 3 tests — regimes, discrete actions, locked-regime mode."""
import pytest
import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.regime import (
    NOISE, BULL, BEAR, N_REGIMES,
    TRANSITION_MATRIX, RegimeStepParams,
    transition_regime, get_regime_params,
)
from lob_sim.actions import (
    ACTION_TABLE, N_ACTIONS,
    action_index_to_offsets, offsets_to_action_index,
)
from lob_sim.step import make_step_fn, run_episode


class TestRegime:
    def test_transition_matrix_rows_sum_to_one(self):
        """Each row of TRANSITION_MATRIX sums to 1.0."""
        for i in range(N_REGIMES):
            assert float(jnp.sum(TRANSITION_MATRIX[i])) == pytest.approx(1.0)

    def test_transition_stays_mostly(self):
        """Over 1000 transitions from NOISE, >90% should stay NOISE."""
        keys = jax.random.split(jax.random.PRNGKey(0), 1000)
        regimes = jax.vmap(lambda k: transition_regime(jnp.int32(NOISE), k))(keys)
        noise_count = int(jnp.sum(regimes == NOISE))
        assert noise_count > 900, f"Only {noise_count}/1000 stayed NOISE"

    def test_all_regimes_reachable(self):
        """Over 10000 transitions from NOISE, all 3 regimes should appear."""
        keys = jax.random.split(jax.random.PRNGKey(42), 10000)
        regimes = jax.vmap(lambda k: transition_regime(jnp.int32(NOISE), k))(keys)
        unique = set(int(r) for r in regimes)
        assert unique == {0, 1, 2}, f"Only reached regimes {unique}"

    def test_get_regime_params_noise(self):
        """get_regime_params(NOISE) returns market_buy_prob=0.15."""
        params = get_regime_params(NOISE)
        assert float(params.market_buy_prob) == pytest.approx(0.15)

    def test_get_regime_params_bull(self):
        """get_regime_params(BULL) returns market_buy_prob=0.30, market_sell_prob=0.05."""
        params = get_regime_params(BULL)
        assert float(params.market_buy_prob) == pytest.approx(0.30)
        assert float(params.market_sell_prob) == pytest.approx(0.05)

    def test_jit_compatible(self):
        """transition_regime and get_regime_params work under jit."""
        jit_transition = jax.jit(transition_regime)
        r = jit_transition(jnp.int32(NOISE), jax.random.PRNGKey(0))
        assert int(r) in {0, 1, 2}

        jit_params = jax.jit(get_regime_params)
        p = jit_params(jnp.int32(BULL))
        assert float(p.market_buy_prob) == pytest.approx(0.30)


class TestActions:
    def test_table_shape(self):
        """ACTION_TABLE has shape (N_ACTIONS, 2)."""
        assert ACTION_TABLE.shape == (N_ACTIONS, 2)

    def test_table_contents(self):
        """ACTION_TABLE[0] = [1,1], ACTION_TABLE[-1] = [5,5]."""
        assert int(ACTION_TABLE[0, 0]) == 1
        assert int(ACTION_TABLE[0, 1]) == 1
        assert int(ACTION_TABLE[N_ACTIONS - 1, 0]) == 5
        assert int(ACTION_TABLE[N_ACTIONS - 1, 1]) == 5

    def test_round_trip(self):
        """offsets_to_action_index(action_index_to_offsets(i)) == i for all i."""
        for i in range(N_ACTIONS):
            offsets = action_index_to_offsets(i)
            idx = offsets_to_action_index(int(offsets[0]), int(offsets[1]))
            assert int(idx) == i, f"Round-trip failed for action {i}"


class TestLockedRegime:
    def test_locked_noise_stays_noise(self):
        """With locked_regime=0, regime is 0 for all T steps."""
        config = SimConfig()
        T = 500
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        _, outputs = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=NOISE)
        assert jnp.all(outputs['regime'] == NOISE)

    def test_locked_bull_stays_bull(self):
        """With locked_regime=1, regime is 1 for all T steps."""
        config = SimConfig()
        T = 500
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        _, outputs = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=BULL)
        assert jnp.all(outputs['regime'] == BULL)

    def test_unlocked_switches(self):
        """With locked_regime=-1, at least 2 different regimes appear over T=5000 steps."""
        config = SimConfig()
        T = 5000
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        _, outputs = run_episode(config, jax.random.PRNGKey(42), actions, locked_regime=-1)
        unique = jnp.unique(outputs['regime'])
        assert len(unique) >= 2, f"Only saw regimes: {unique}"


class TestRegimeEffectOnDynamics:
    def test_bull_drifts_up(self):
        """In locked bull regime, mean mid-price change > 0 over T=2000 steps."""
        config = SimConfig()
        T = 2000
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        _, outputs = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=BULL)
        assert float(outputs['mid_price'][-1]) > float(outputs['mid_price'][0])

    def test_bear_drifts_down(self):
        """In locked bear regime, mean mid-price change < 0 over T=2000 steps."""
        config = SimConfig()
        T = 2000
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        _, outputs = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=BEAR)
        assert float(outputs['mid_price'][-1]) < float(outputs['mid_price'][0])

    def test_noise_no_drift(self):
        """In locked noise regime, abs(mid_prices[-1] - mid_prices[0]) < bull drift."""
        config = SimConfig()
        T = 2000
        actions = jnp.full((T,), 4, dtype=jnp.int32)

        _, bull_out = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=BULL)
        bull_drift = abs(float(bull_out['mid_price'][-1]) - float(bull_out['mid_price'][0]))

        _, noise_out = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=NOISE)
        noise_drift = abs(float(noise_out['mid_price'][-1]) - float(noise_out['mid_price'][0]))

        assert noise_drift < bull_drift

    def test_bull_more_ask_fills(self):
        """In bull regime, total ask fills > total bid fills (more market buys)."""
        config = SimConfig()
        T = 2000
        actions = jnp.full((T,), 0, dtype=jnp.int32)  # action 0 = (1,1), close to best
        _, outputs = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=BULL)
        total_ask = float(jnp.sum(outputs['ask_fill']))
        total_bid = float(jnp.sum(outputs['bid_fill']))
        assert total_ask > total_bid, f"ask_fills={total_ask:.2f} <= bid_fills={total_bid:.2f}"

    def test_bear_more_bid_fills(self):
        """In bear regime, total bid fills > total ask fills."""
        config = SimConfig()
        T = 2000
        actions = jnp.full((T,), 0, dtype=jnp.int32)  # action 0 = (1,1), close to best
        _, outputs = run_episode(config, jax.random.PRNGKey(0), actions, locked_regime=BEAR)
        total_ask = float(jnp.sum(outputs['ask_fill']))
        total_bid = float(jnp.sum(outputs['bid_fill']))
        assert total_bid > total_ask, f"bid_fills={total_bid:.2f} <= ask_fills={total_ask:.2f}"

    def test_run_episode_with_discrete_action(self):
        """run_episode works with scalar int32 actions."""
        config = SimConfig()
        T = 100
        actions = jnp.full((T,), 4, dtype=jnp.int32)  # action 4 = (3,3)
        final, outputs = run_episode(config, jax.random.PRNGKey(0), actions)
        assert final.step_count == T
        assert outputs['mid_price'].shape == (T,)
        assert outputs['regime'].shape == (T,)
