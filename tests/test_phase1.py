"""Phase 1 tests — core book mechanics."""
import pytest
import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.state import OrderBookState, init_state
from lob_sim.matching import fill_market_buy, fill_market_sell
from lob_sim.background import generate_background_flow


class TestInitState:
    def test_volumes_are_decreasing(self):
        """Volume at level 0 > volume at level 10 (exponential decay)."""
        config = SimConfig()
        state = init_state(config, jax.random.PRNGKey(0))
        assert float(state.bid_volumes[0]) > float(state.bid_volumes[10])
        assert float(state.ask_volumes[0]) > float(state.ask_volumes[10])

    def test_mid_price_is_100(self):
        """Starting mid-price should be 100.0."""
        config = SimConfig()
        state = init_state(config, jax.random.PRNGKey(0))
        assert float(state.mid_price) == pytest.approx(100.0)

    def test_shapes_correct(self):
        """bid_volumes and ask_volumes have shape (n_levels,)."""
        config = SimConfig()
        state = init_state(config, jax.random.PRNGKey(0))
        assert state.bid_volumes.shape == (config.n_levels,)
        assert state.ask_volumes.shape == (config.n_levels,)

    def test_all_fields_are_jax_arrays(self):
        """Every field in OrderBookState must be a jnp.ndarray."""
        config = SimConfig()
        state = init_state(config, jax.random.PRNGKey(0))
        for field_name in OrderBookState._fields:
            val = getattr(state, field_name)
            assert isinstance(val, jax.Array), f"{field_name} is not a jax array"


class TestMatchingEngine:
    def _make_book(self, n=100, vol=5.0):
        vols = jnp.full(n, vol)
        return vols, jnp.int32(1), jnp.float32(100.0), jnp.float32(0.01)

    def test_partial_fill_single_level(self):
        """Fill qty=3 against ask_volumes=[5, 5, ...]. Level 0: 5->2, filled=3."""
        ask_vols, hs, mid, tick = self._make_book()
        new_ask, filled, avg_price, new_mid, new_hs = fill_market_buy(
            ask_vols, jnp.float32(3.0), hs, mid, tick
        )
        assert float(filled) == pytest.approx(3.0)
        assert float(new_ask[0]) == pytest.approx(2.0)
        assert float(new_mid) == pytest.approx(float(mid))

    def test_full_level_depletion(self):
        """Fill qty=5 against ask_volumes=[5, 5, ...]. Level 0 fully consumed."""
        ask_vols, hs, mid, tick = self._make_book()
        new_ask, filled, avg_price, new_mid, new_hs = fill_market_buy(
            ask_vols, jnp.float32(5.0), hs, mid, tick
        )
        assert float(filled) == pytest.approx(5.0)
        # After roll, new level 0 should be 5.0 (was level 1)
        assert float(new_ask[0]) == pytest.approx(5.0)

    def test_multi_level_fill(self):
        """Fill qty=8 against ask_volumes=[5, 5, ...]. Depletes level 0, partially fills level 1."""
        ask_vols, hs, mid, tick = self._make_book()
        new_ask, filled, avg_price, new_mid, new_hs = fill_market_buy(
            ask_vols, jnp.float32(8.0), hs, mid, tick
        )
        assert float(filled) == pytest.approx(8.0)
        # Level 0 was depleted (had 5), level 1 had 5, 3 consumed -> 2 left
        # After roll by 1, new level 0 = 2.0
        assert float(new_ask[0]) == pytest.approx(2.0)

    def test_mid_price_shifts_on_depletion(self):
        """When best ask is depleted, mid_price must increase."""
        ask_vols, hs, mid, tick = self._make_book()
        _, _, _, new_mid, _ = fill_market_buy(
            ask_vols, jnp.float32(5.0), hs, mid, tick
        )
        assert float(new_mid) > float(mid)

    def test_vwap_correct(self):
        """VWAP fill price matches hand-calculated value."""
        n = 100
        ask_vols = jnp.zeros(n).at[0].set(3.0).at[1].set(5.0)
        hs = jnp.int32(1)
        mid = jnp.float32(100.0)
        tick = jnp.float32(0.01)
        # Price at level 0 = 100.0 + 1 * 0.01 = 100.01
        # Price at level 1 = 100.0 + 2 * 0.01 = 100.02
        # Fill qty=6: 3 at 100.01, 3 at 100.02
        _, filled, avg_price, _, _ = fill_market_buy(
            ask_vols, jnp.float32(6.0), hs, mid, tick
        )
        expected = (3.0 * 100.01 + 3.0 * 100.02) / 6.0
        assert float(filled) == pytest.approx(6.0)
        assert float(avg_price) == pytest.approx(expected, rel=1e-5)

    def test_roll_zero_fill(self):
        """After jnp.roll, deep end must be zeros, not stale wrapped values."""
        n = 100
        # First 3 levels have volume, rest is 1.0 for easy checking
        ask_vols = jnp.ones(n)
        ask_vols = ask_vols.at[0].set(2.0).at[1].set(2.0).at[2].set(2.0)
        hs = jnp.int32(1)
        mid = jnp.float32(100.0)
        tick = jnp.float32(0.01)
        # Fill 6.0 to deplete levels 0,1,2
        new_ask, filled, _, _, _ = fill_market_buy(
            ask_vols, jnp.float32(6.0), hs, mid, tick
        )
        assert float(filled) == pytest.approx(6.0)
        # Last 3 levels must be zero (not stale wrapped values)
        assert float(new_ask[-1]) == pytest.approx(0.0)
        assert float(new_ask[-2]) == pytest.approx(0.0)
        assert float(new_ask[-3]) == pytest.approx(0.0)

    def test_sell_mirrors_buy(self):
        """fill_market_sell should be symmetric to fill_market_buy."""
        n = 100
        vols = jnp.full(n, 5.0)
        hs = jnp.int32(1)
        mid = jnp.float32(100.0)
        tick = jnp.float32(0.01)
        qty = jnp.float32(5.0)

        _, _, _, buy_mid, _ = fill_market_buy(vols, qty, hs, mid, tick)
        _, _, _, sell_mid, _ = fill_market_sell(vols, qty, hs, mid, tick)
        # Buy pushes mid up, sell pushes mid down
        assert float(buy_mid) > float(mid)
        assert float(sell_mid) < float(mid)

    def test_zero_qty_noop(self):
        """Filling qty=0 should not change anything."""
        ask_vols = jnp.full(100, 5.0)
        hs = jnp.int32(1)
        mid = jnp.float32(100.0)
        tick = jnp.float32(0.01)
        new_ask, filled, avg_price, new_mid, new_hs = fill_market_buy(
            ask_vols, jnp.float32(0.0), hs, mid, tick
        )
        assert float(filled) == pytest.approx(0.0)
        assert float(new_mid) == pytest.approx(float(mid))
        assert jnp.allclose(new_ask, ask_vols)

    def test_jit_compatible(self):
        """jax.jit(fill_market_buy) runs without error."""
        ask_vols = jnp.full(100, 5.0)
        hs = jnp.int32(1)
        mid = jnp.float32(100.0)
        tick = jnp.float32(0.01)
        jit_fn = jax.jit(fill_market_buy)
        new_ask, filled, avg_price, new_mid, new_hs = jit_fn(
            ask_vols, jnp.float32(3.0), hs, mid, tick
        )
        assert float(filled) == pytest.approx(3.0)


class TestBackgroundFlow:
    def test_market_orders_fire_at_expected_rate(self):
        """Over 1000 calls, market buys should fire ~15% of the time (within 10-20%)."""
        config = SimConfig()
        n = config.n_levels
        vols = jnp.full(n, 5.0)
        count = 0
        keys = jax.random.split(jax.random.PRNGKey(42), 1000)
        for i in range(1000):
            _, _, mbq, _ = generate_background_flow(vols, vols, config, keys[i])
            if float(mbq) > 0:
                count += 1
        assert 100 <= count <= 250, f"market buy fired {count}/1000 times"

    def test_cancellations_reduce_volume(self):
        """After background flow, total volume should generally decrease (cancellations)."""
        config = SimConfig(market_buy_prob=0.0, market_sell_prob=0.0, limit_order_rate=0.0)
        n = config.n_levels
        vols = jnp.full(n, 10.0)
        total_before = float(jnp.sum(vols)) * 2
        total_after_sum = 0.0
        keys = jax.random.split(jax.random.PRNGKey(7), 100)
        for i in range(100):
            bv, av, _, _ = generate_background_flow(vols, vols, config, keys[i])
            total_after_sum += float(jnp.sum(bv) + jnp.sum(av))
        avg_after = total_after_sum / 100
        assert avg_after < total_before

    def test_new_limit_orders_add_volume(self):
        """Limit order arrivals add volume, more at near levels than deep."""
        config = SimConfig(cancel_prob=0.0, market_buy_prob=0.0, market_sell_prob=0.0)
        n = config.n_levels
        vols = jnp.zeros(n)
        near_additions = 0.0
        deep_additions = 0.0
        keys = jax.random.split(jax.random.PRNGKey(99), 500)
        for i in range(500):
            bv, av, _, _ = generate_background_flow(vols, vols, config, keys[i])
            near_additions += float(jnp.sum(bv[:10]) + jnp.sum(av[:10]))
            deep_additions += float(jnp.sum(bv[90:]) + jnp.sum(av[90:]))
        assert near_additions > deep_additions

    def test_rng_keys_independent(self):
        """Different rng keys produce different market order quantities."""
        config = SimConfig()
        n = config.n_levels
        vols = jnp.full(n, 5.0)
        _, _, mbq1, msq1 = generate_background_flow(vols, vols, config, jax.random.PRNGKey(0))
        _, _, mbq2, msq2 = generate_background_flow(vols, vols, config, jax.random.PRNGKey(1))
        # At least one of the four values should differ
        all_same = (
            float(mbq1) == float(mbq2)
            and float(msq1) == float(msq2)
        )
        # Extremely unlikely to be identical with different keys
        # But to be safe, test with more keys
        differs = False
        for k in range(10):
            _, _, a, b = generate_background_flow(vols, vols, config, jax.random.PRNGKey(k))
            _, _, c, d = generate_background_flow(vols, vols, config, jax.random.PRNGKey(k + 100))
            if float(a) != float(c) or float(b) != float(d):
                differs = True
                break
        assert differs

    def test_jit_compatible(self):
        """jax.jit(generate_background_flow) runs without error."""
        config = SimConfig()
        n = config.n_levels
        vols = jnp.full(n, 5.0)
        jit_fn = jax.jit(generate_background_flow, static_argnums=(2,))
        bv, av, mbq, msq = jit_fn(vols, vols, config, jax.random.PRNGKey(0))
        assert bv.shape == (n,)
        assert av.shape == (n,)
