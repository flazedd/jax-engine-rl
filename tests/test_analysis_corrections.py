"""Regression checks for the thesis analysis audit."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from agents.rl2 import RL2Agent
from evaluation.belief_swap import action_probs_with_belief

@pytest.mark.parametrize('integration', ['concat', 'hypernet'])
@pytest.mark.parametrize('simplex,layernorm', [(False,False),(True,False),(False,True)])
def test_rl2_substitution_identity(integration, simplex, layernorm):
    agent = RL2Agent(obs_size=16, n_actions=3, hidden_dim=8,
        integration=integration, policy_obs_dim=11, concat_policy_reads_obs=True,
        policy_trunk_layers=1, policy_trunk_hidden=7,
        belief_simplex_head=simplex, belief_layernorm=layernorm)
    model = agent._model()
    obs = jax.random.normal(jax.random.PRNGKey(8), (5,16))
    carry = jax.random.normal(jax.random.PRNGKey(9), (5,8))
    variables = model.init(jax.random.PRNGKey(1), carry[0], obs[0])
    updated, logits, _, _ = jax.vmap(lambda h,o:model.apply(variables,h,o))(carry,obs)
    actual = action_probs_with_belief(agent, {'params':variables}, np.asarray(obs),
                                    np.asarray(updated), 'rl2')
    np.testing.assert_allclose(actual, jax.nn.softmax(logits), atol=2e-7)
    changed = action_probs_with_belief(agent, {'params':variables}, np.asarray(obs),
                                     np.asarray(updated)[::-1], 'rl2')
    assert np.max(np.abs(actual-changed)) > 1e-8

from scripts.m5r_refresh_probe_returns import evaluation_returns_for_seeds, refresh_probe
from scripts.m5r_hypothesis_tests import _within_variant_correlation

def test_seed_join_uses_ids_and_rejects_missing_or_duplicate_ids():
    methods={'m':{'experiment':'e','seeds':[0,1,10,2],
                  'per_seed_evaluation_return':[100,101,110,102]}}
    assert evaluation_returns_for_seeds(methods,'m','e',[2,10,1,0]) == [102,110,101,100]
    with pytest.raises(ValueError): evaluation_returns_for_seeds(methods,'m','wrong',[0,1,10,2])
    with pytest.raises(ValueError): evaluation_returns_for_seeds(methods,'m','e',[0,1,2])
    methods['m']['seeds']=[0,1,1,2]
    with pytest.raises(ValueError): evaluation_returns_for_seeds(methods,'m','e',[0,1,1,2])

def test_within_variant_correlation_removes_offsets_and_keeps_run_pair():
    points=[{'method':m,'seed':s,'error':offset+s,
             'gap_closed':100*offset-2*s} for m,offset in [('a',0),('b',10)] for s in range(8)]
    result=_within_variant_correlation(points,'error',n_boot=100)
    assert result['correlation'] == pytest.approx(-1)
    np.testing.assert_allclose(result['ci'],[-1,-1])
    relabelled=[dict(p,seed=100-p['seed']) for p in points]
    assert _within_variant_correlation(relabelled,'error',n_boot=100)['ci'] == result['ci']


def test_temporal_intervals_smooth_before_quantiles():
    from plotting.m5r_plots import _mean_curve_ci, _independent_delta_ci
    # Each run has the same two-step mean despite opposite raw trajectories.
    # Any whole-run bootstrap must therefore give a degenerate smoothed CI.
    curves = np.array([[0., 2.], [2., 0.]])
    mean, lo, hi = _mean_curve_ci(curves, smoothing_window=3, n_boot=200)
    np.testing.assert_allclose([mean, lo, hi], np.ones((3, 2)))
    mean, lo, hi = _independent_delta_ci(
        2 * curves, curves, n_boot=200, smoothing_window=3)
    np.testing.assert_allclose([mean, lo, hi], np.ones((3, 2)))


def test_policy_heatmap_keeps_boundary_ties():
    from envs.market_making_v1 import MarketMakingV1
    from oracles.value_iteration import solve_value_iteration
    from plotting.m2_plots import _tied_optimal_actions
    from training.config import _load_yaml_with_extends
    from pathlib import Path
    cfg = _load_yaml_with_extends(Path(__file__).resolve().parents[1] /
                                 'experiments/configs/m5r_e9/oracle_ppo.yaml')
    vi = solve_value_iteration(MarketMakingV1(**cfg['env']['params']))
    tied = _tied_optimal_actions(vi.Q)
    np.testing.assert_array_equal(tied[0, 1], [True, False, True])
    np.testing.assert_array_equal(tied[0, 2], tied[0, 1])
    np.testing.assert_array_equal(np.flatnonzero(np.any(tied[:, 1] != tied[:, 2], axis=-1)),
                                  [4, 6])  # q = -1, +1
