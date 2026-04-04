"""Step function and episode runner."""
import jax
import jax.numpy as jnp

from lob_sim.config import SimConfig
from lob_sim.state import OrderBookState, init_state
from lob_sim.matching import fill_market_buy, fill_market_sell
from lob_sim.background import generate_background_flow
from lob_sim.obs import observe
from lob_sim.regime import transition_regime, get_regime_params
from lob_sim.actions import ACTION_TABLE


def make_step_fn(config: SimConfig, locked_regime=-1):
    """Create a step function closed over config.

    Returns step_fn(state, action) -> (new_state, output_dict)
    where action is either:
        - scalar int32 (Phase 3+): index into ACTION_TABLE (0 to N_ACTIONS-1)
        - array shape (2,) (Phase 1-2): [bid_offset_ticks, ask_offset_ticks]
    """
    n = config.n_levels
    tick = config.tick_size
    agent_size = config.agent_order_size
    _levels = jnp.arange(n)

    def step_fn(state, action):
        next_state, inner_out = _step_inner(state, action)
        # If already done at start of step, keep state unchanged
        next_state = jax.tree.map(
            lambda old, new: jnp.where(state.done, old, new),
            state, next_state,
        )
        obs = observe(next_state, config)
        reward = jnp.where(state.done, 0.0, inner_out['reward'])

        # Compute actual spread from volume gaps
        bid_nonzero = next_state.bid_volumes > 0
        ask_nonzero = next_state.ask_volumes > 0
        bid_gap = jnp.where(jnp.any(bid_nonzero), jnp.argmax(bid_nonzero), jnp.int32(0))
        ask_gap = jnp.where(jnp.any(ask_nonzero), jnp.argmax(ask_nonzero), jnp.int32(0))
        hs_float = next_state.half_spread_ticks.astype(jnp.float32)
        spread = (2.0 * hs_float + bid_gap.astype(jnp.float32) + ask_gap.astype(jnp.float32)) * tick

        return next_state, {
            'obs': obs,
            'reward': reward,
            'mid_price': next_state.mid_price,
            'spread': spread,
            'inventory': next_state.inventory,
            'cash': next_state.cash,
            'bid_fill': jnp.where(state.done, 0.0, inner_out['bid_fill']),
            'ask_fill': jnp.where(state.done, 0.0, inner_out['ask_fill']),
            'done': next_state.done,
            'bid_volumes': next_state.bid_volumes[:20],
            'ask_volumes': next_state.ask_volumes[:20],
            'regime': next_state.regime,
        }

    def _step_inner(state, action):
        mid = state.mid_price
        hs = state.half_spread_ticks

        if action.ndim == 0:
            # Phase 3+: discrete action + regime system
            key, k1, k_regime = jax.random.split(state.rng_key, 3)
            offsets = ACTION_TABLE[action]
            bid_level = jnp.clip(offsets[0].astype(jnp.int32), 0, n - 1)
            ask_level = jnp.clip(offsets[1].astype(jnp.int32), 0, n - 1)

            # Regime transition
            new_regime = transition_regime(state.regime, k_regime)
            new_regime = jnp.where(locked_regime >= 0, locked_regime, new_regime)
            regime_params = get_regime_params(new_regime)

            # Apply price drift
            mid = mid + regime_params.price_drift
        else:
            # Phase 1-2: continuous (2,) action, no regime system
            key, k1 = jax.random.split(state.rng_key)
            bid_level = jnp.clip(action[0].astype(jnp.int32), 0, n - 1)
            ask_level = jnp.clip(action[1].astype(jnp.int32), 0, n - 1)
            new_regime = state.regime
            regime_params = None

        # 1. Place agent orders
        bids = state.bid_volumes.at[bid_level].add(agent_size)
        asks = state.ask_volumes.at[ask_level].add(agent_size)

        # 2-3. Generate background flow
        bids, asks, mkt_buy, mkt_sell = generate_background_flow(
            bids, asks, config, k1, regime_params=regime_params
        )

        # Pre-match snapshot at agent levels
        pre_bid = bids[bid_level]
        pre_ask = asks[ask_level]

        # 4. Compute consumed at agent levels (same formula as matching engine)
        ask_cs_prev = jnp.concatenate([jnp.zeros(1), jnp.cumsum(asks)[:-1]])
        ask_consumed = jnp.minimum(asks, jnp.maximum(0.0, mkt_buy - ask_cs_prev))

        bid_cs_prev = jnp.concatenate([jnp.zeros(1), jnp.cumsum(bids)[:-1]])
        bid_consumed = jnp.minimum(bids, jnp.maximum(0.0, mkt_sell - bid_cs_prev))

        # 5. Pro-rata agent fills
        ask_fill = jnp.where(
            pre_ask > 0,
            ask_consumed[ask_level] / pre_ask * agent_size,
            0.0,
        )
        ask_fill = jnp.minimum(ask_fill, agent_size)
        bid_fill = jnp.where(
            pre_bid > 0,
            bid_consumed[bid_level] / pre_bid * agent_size,
            0.0,
        )
        bid_fill = jnp.minimum(bid_fill, agent_size)

        # 6. Match market orders (updates volumes and mid-price)
        new_asks, _, _, mid1, hs1 = fill_market_buy(
            asks, mkt_buy, hs, mid, tick
        )
        ask_shift = jnp.round((mid1 - mid) / tick).astype(jnp.int32)

        # Cross-side: push bids deeper to account for ask-side mid shift
        adjusted_bids = jnp.roll(bids, ask_shift)
        adjusted_bids = jnp.where(_levels < ask_shift, 0.0, adjusted_bids)

        new_bids, _, _, mid2, hs2 = fill_market_sell(
            adjusted_bids, mkt_sell, hs1, mid1, tick
        )
        bid_shift = jnp.round((mid1 - mid2) / tick).astype(jnp.int32)

        # 7. Remove unfilled agent volume (accounting for shift/roll)
        # Ask side: matching rolled by -ask_shift
        shifted_ask = ask_level - ask_shift
        valid_ask = (shifted_ask >= 0).astype(jnp.float32)
        safe_ask = jnp.clip(shifted_ask, 0, n - 1)
        new_asks = new_asks.at[safe_ask].add(-(agent_size - ask_fill) * valid_ask)
        new_asks = jnp.maximum(new_asks, 0.0)

        # Bid side: cross-side pushed by +ask_shift, then matching rolled by -bid_shift
        shifted_bid = bid_level + ask_shift - bid_shift
        valid_bid = (shifted_bid >= 0).astype(jnp.float32)
        safe_bid = jnp.clip(shifted_bid, 0, n - 1)
        new_bids = new_bids.at[safe_bid].add(-(agent_size - bid_fill) * valid_bid)
        new_bids = jnp.maximum(new_bids, 0.0)

        # 8. Cross-side: push asks deeper to account for bid-side mid shift
        new_asks = jnp.roll(new_asks, bid_shift)
        new_asks = jnp.where(_levels < bid_shift, 0.0, new_asks)

        # 9. Update inventory and cash
        bid_price = state.mid_price - (bid_level.astype(jnp.float32) + hs.astype(jnp.float32)) * tick
        ask_price = state.mid_price + (ask_level.astype(jnp.float32) + hs.astype(jnp.float32)) * tick

        new_inventory = jnp.clip(state.inventory + bid_fill - ask_fill,
                                 -config.max_inventory, config.max_inventory)
        new_cash = state.cash - bid_fill * bid_price + ask_fill * ask_price
        new_step = state.step_count + 1

        done = new_step >= config.max_steps

        new_state = OrderBookState(
            bid_volumes=new_bids,
            ask_volumes=new_asks,
            mid_price=mid2,
            half_spread_ticks=hs2,
            regime=new_regime,
            inventory=new_inventory,
            cash=new_cash,
            step_count=new_step,
            done=done,
            rng_key=key,
        )

        # Spread capture + mark-to-market on existing inventory
        bid_edge = (bid_level.astype(jnp.float32) + hs.astype(jnp.float32)) * tick
        ask_edge = (ask_level.astype(jnp.float32) + hs.astype(jnp.float32)) * tick
        mtm = state.inventory * (mid2 - state.mid_price)
        reward = bid_fill * bid_edge + ask_fill * ask_edge + mtm

        return new_state, {
            'reward': reward,
            'bid_fill': bid_fill,
            'ask_fill': ask_fill,
        }

    return step_fn


def run_episode(config, rng_key, actions, locked_regime=-1):
    """Run a full episode using lax.scan.

    Args:
        config: SimConfig
        rng_key: JAX PRNGKey
        actions: array of shape (T,) int32 or (T, 2) float
        locked_regime: If >= 0, lock the regime to this value. -1 for free transitions.

    Returns:
        (final_state, outputs) where outputs is a dict of arrays with leading dim T.
    """
    state = init_state(config, rng_key)
    step_fn = make_step_fn(config, locked_regime=locked_regime)
    final_state, outputs = jax.lax.scan(step_fn, state, actions)
    return final_state, outputs
