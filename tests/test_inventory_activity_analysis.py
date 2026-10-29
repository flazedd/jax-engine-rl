"""Check accepted-fill accounting against actual boundary and simultaneous-fill outcomes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envs.market_making_v1 import MarketMakingV1
from scripts.inventory_activity_analysis import accepted_fills


@pytest.mark.parametrize("raw_bid,raw_ask", [(0,0),(0,1),(1,0),(1,1)])
def test_accepted_fills_including_simultaneous_draws_and_inventory_bounds(raw_bid, raw_ask):
    # Force all four raw outcomes and let the real environment reject blocked sides.
    env=MarketMakingV1(n_regimes=2, transition_matrix=(1.,0.,0.,1.),
        initial_distribution=(1.,0.), lock_regime=0, inventory_penalty=.05,
        regime_p_tight_bid=(float(raw_bid),)*2, regime_p_wide_bid=(float(raw_bid),)*2,
        regime_p_tight_ask=(float(raw_ask),)*2, regime_p_wide_ask=(float(raw_ask),)*2)
    for q in [-5,0,5]:
        for action in range(3):
            state={'q':jnp.asarray(q),'t':jnp.asarray(127),'regime':jnp.asarray(0)}
            _,_,reward,done,info=env.step(state,jnp.asarray(action),jax.random.PRNGKey(17))
            bid,ask=accepted_fills(np.asarray(action),np.asarray(q),np.asarray(info['inventory']),np.asarray(reward),.05)
            assert int(bid)==int(raw_bid and q<5)
            assert int(ask)==int(raw_ask and q>-5)
            assert bool(done) # post inventory must survive terminal auto-reset in info.
