"""Timing, repeated switches, terminal penalties, and seed-level uncertainty."""
import numpy as np
import pytest

from scripts.natural_switch_evaluation import summarize_rollout


def test_initial_regime_is_not_switch_and_repeated_switch_resets_window():
    # Episode 0 switches at t=1 and t=3; episode 1 never switches.
    regime = np.array([[0, 2], [1, 2], [1, 2], [0, 2], [0, 2]])
    reward = np.array([[100, 100], [2, 100], [4, 100], [6, 100], [8, 100]])
    inventory = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    done = np.zeros_like(regime); done[-1] = 1
    result = summarize_rollout(regime, reward, inventory, done, .05)
    first, later = result['bins'][:2]
    assert result['switches'] == 2
    assert result['episodes_without_switch'] == 1
    assert result['episodes_with_multiple_switches'] == 1
    assert (first['steps'], first['episodes'], first['switches']) == (2, 1, 2)
    assert first['mean_reward'] == 4
    assert first['mean_inventory_penalty'] == pytest.approx(.25)
    assert (later['steps'], later['episodes'], later['switches']) == (2, 1, 2)
    assert later['mean_reward'] == 6
    # Terminal q=4 is counted before reset, contributing penalty .8.
    assert later['mean_inventory_penalty'] == pytest.approx(.5)
    assert sum(b['steps'] for b in result['bins']) == 4
    assert result['bins'][-1]['mean_reward'] is None


def test_switch_at_final_decision_counts_only_age_zero():
    regime = np.array([[0], [0], [1]])
    done = np.array([[0], [0], [1]])
    result = summarize_rollout(regime, np.ones((3, 1)), np.ones((3, 1)), done, .05)
    assert result['bins'][0]['steps'] == 1
    assert result['bins'][1]['steps'] == 0


def test_internal_episode_reset_is_rejected():
    with pytest.raises(ValueError, match='internal resets'):
        summarize_rollout(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)),
                          np.array([[0], [1], [1]]), .05)


def test_seed_means_are_equal_weighted_and_differences_pair_seeds():
    from scripts.summarize_natural_switches import estimate
    # The statistic averages runs equally, irrespective of episode/step counts.
    values = np.array([0., 2.])
    indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert estimate(values, indices)['mean'] == 1
    assert estimate(values + 3 - values, indices)['ci95'] == [3., 3.]


def test_collector_preserves_post_fill_inventory_on_terminal_step():
    import jax
    import jax.numpy as jnp
    from envs.market_making_v1 import MarketMakingV1
    from evaluation.action_distribution import collect_action_regime_rollouts

    class AlwaysSymmetric:
        def act(self, state, obs, key):
            return jnp.array(0), {}, None

    env = MarketMakingV1(episode_length=1, n_regimes=2,
                         transition_matrix=(1., 0., 0., 1.),
                         initial_distribution=(1., 0.),
                         regime_p_tight_bid=(1., 1.), regime_p_wide_bid=(1., 1.),
                         regime_p_tight_ask=(0., 0.), regime_p_wide_ask=(0., 0.),
                         inventory_penalty=.05)
    out = collect_action_regime_rollouts(env, AlwaysSymmetric(), {}, 2, 1, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(out['inventory'], [[0, 0]])
    np.testing.assert_array_equal(out['post_inventory'], [[1, 1]])
    np.testing.assert_array_equal(out['done'], [[1, 1]])
    np.testing.assert_allclose(out['reward'], [[.95, .95]])


def test_summary_keeps_seed_weights_and_rejects_duplicate_runs():
    from scripts.natural_switch_evaluation import BINS, CELLS
    from scripts.summarize_natural_switches import summarize
    rows = []
    for cell in CELLS:
        for seed in range(20):
            value = seed + (2 if cell.endswith('hypernet') else 0)
            rows.append({'method': cell, 'seed': seed, 'episodes': 512,
                         'switches': 300, 'episodes_without_switch': 250,
                         'episodes_with_one_switch': 224, 'episodes_with_multiple_switches': 38,
                         'mean_episode_return': 1., 'published_return': 1.,
                         'bins': [{'interval': list(b), 'episodes': 10, 'switches': 10,
                                   'steps': 10 * (seed + 1), 'mean_reward': value,
                                   'mean_inventory_penalty': .1} for b in BINS]})
    payload = {'identity': {'bins': [list(b) for b in BINS]}, 'rows': rows}
    result = summarize(payload)
    assert result['methods']['rl2_concat']['bins'][0]['mean_reward']['mean'] == 9.5
    assert result['methods']['rl2_concat']['bins'][0]['steps'] == 2100
    assert result['differences']['rl2'][0]['mean_reward']['ci95'] == [2., 2.]
    payload['rows'][-1] = payload['rows'][0]
    with pytest.raises(ValueError, match='distinct seeds'):
        summarize(payload)
