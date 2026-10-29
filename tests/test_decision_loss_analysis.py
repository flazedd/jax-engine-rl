import numpy as np
from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import _transition_tables,finite_horizon_optimum
from scripts.decision_loss_analysis import finite_values

def test_backward_values_match_existing_oracle_and_loss_telescopes():
 env=MarketMakingV1(n_regimes=1,inventory_max=5,episode_length=128,initial_distribution=(1.,))
 V,Q,L=finite_values(env)
 np.testing.assert_allclose(V[0,5,0],finite_horizon_optimum(env),rtol=0,atol=1e-10)
 P,R=_transition_tables(env);d=np.zeros((11,1));d[5,0]=1;total_reward=0.;total_loss=0.
 # Fixed symmetric policy: its exact loss sum must equal its oracle value gap.
 for t in range(128):
  total_reward+=np.sum(d*R[:,:,0]);total_loss+=np.sum(d*L[t,:,:,0]);d=np.einsum('iz,izjk->jk',d,P[:,:,0])
 np.testing.assert_allclose(V[0,5,0]-total_reward,total_loss,rtol=0,atol=1e-10)
 np.testing.assert_allclose(L.min(-1),0,atol=1e-12)
