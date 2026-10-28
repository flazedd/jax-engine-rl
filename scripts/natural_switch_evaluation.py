"""Replay frozen main policies and describe trading after natural regime changes.

Uses the original evaluation episodes and verifies total returns against the
published values. No retraining, forced switches, or altered environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from evaluation.action_distribution import collect_action_regime_rollouts
from evaluation.posterior_probe import load_experiment
from scripts.post_training_evaluation import METHODS, N_EPISODES, EPISODE_LENGTH
from scripts.prepare_results import digest
from utils.paths import analysis_dir, experiment_dir

ROOT = Path(__file__).resolve().parents[1]
CELLS = ('rl2_concat', 'rl2_hypernet', 'varibad_concat', 'varibad_hypernet')
BINS = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 39), (40, 79), (80, 126)]


def summarize_rollout(regime, reward, post_inventory, done, penalty_coefficient):
    """Step-weighted means within each seed; windows never cross another switch.

    All columns must be single complete episodes. A switch at t starts a new
    event: r[t] is the first reward under the new regime. Initial regimes and
    terminal transitions after the final trading decision are not switches.
    """
    regime, reward, post_inventory, done = map(
        np.asarray, (regime, reward, post_inventory, done))
    if not (regime.shape == reward.shape == post_inventory.shape == done.shape):
        raise ValueError('Rollout shapes differ')
    if regime.ndim != 2 or np.any(done[:-1]) or not np.all(done[-1]):
        raise ValueError('Expected one complete episode per column without internal resets')
    if not np.isfinite(reward).all() or not np.isfinite(post_inventory).all():
        raise ValueError('Nonfinite rollout values')
    changed = np.zeros(regime.shape, dtype=bool)
    changed[1:] = regime[1:] != regime[:-1]
    times = np.arange(len(regime))[:, None]
    latest = np.maximum.accumulate(np.where(changed, times, -1), axis=0)
    age = np.where(latest >= 0, times - latest, -1)
    # Each event ID is unique across time and episode, even after repeated switches.
    event_id = latest * regime.shape[1] + np.arange(regime.shape[1])[None, :]
    penalties = penalty_coefficient * post_inventory.astype(float) ** 2
    rows = []
    for lo, hi in BINS:
        mask = (age >= lo) & (age <= hi)
        rows.append({
            'interval': [lo, hi], 'steps': int(mask.sum()),
            'episodes': int(mask.any(axis=0).sum()),
            'switches': int(np.unique(event_id[mask]).size),
            'mean_reward': float(reward[mask].mean()) if mask.any() else None,
            'mean_inventory_penalty': float(penalties[mask].mean()) if mask.any() else None,
        })
    switch_counts = changed.sum(axis=0)
    return {
        'episodes': regime.shape[1], 'switches': int(changed.sum()),
        'episodes_without_switch': int((switch_counts == 0).sum()),
        'episodes_with_one_switch': int((switch_counts == 1).sum()),
        'episodes_with_multiple_switches': int((switch_counts > 1).sum()),
        'mean_episode_return': float(reward.astype(float).sum(axis=0).mean()),
        'bins': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=CELLS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or analysis_dir() / 'natural_switch_evaluation.json'
    saved_path = analysis_dir() / 'm5r_post_training_evaluation.json'
    saved = json.loads(saved_path.read_text())['methods']
    identity = {
        'schema_version': 1, 'episodes_per_run': N_EPISODES,
        'episode_length': EPISODE_LENGTH, 'bins': [list(b) for b in BINS],
        'source_sha256': {p: digest(ROOT / p) for p in (
            'scripts/natural_switch_evaluation.py', 'evaluation/action_distribution.py',
            'evaluation/posterior_probe.py', 'envs/market_making_v1.py')},
        'evaluation_reference_sha256': digest(saved_path),
    }
    payload = json.loads(output.read_text()) if output.exists() else {
        'identity': identity, 'rows': [],
        'description': 'Additional descriptive evaluation on original naturally switching episodes; no training.',
        'timing': 'Age 0 is first trading decision under the new regime. Windows end before the next switch or after the last decision.',
        'aggregation': 'Within each seed: mean over eligible steps. Across seeds: equally weighted means, not pooled steps.',
    }
    if payload['identity'] != identity:
        raise ValueError('Existing evaluation has different inputs or code; use a new output path')
    ids = [(r['method'], r['seed']) for r in payload['rows']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate run records')
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, (method, experiment) in enumerate(METHODS):
        if method not in CELLS or args.method not in (None, method):
            continue
        directory = experiment_dir(experiment)
        reference = dict(zip(saved[method]['seeds'], saved[method]['per_seed_evaluation_return']))
        for seed in range(20):
            hashes = {'checkpoint_sha256': digest(directory / f'checkpoint_seed_{seed}.pkl'),
                      'config_sha256': digest(directory / 'config.json')}
            previous = [r for r in payload['rows'] if (r['method'], r['seed']) == (method, seed)]
            if previous:
                if any(previous[0][k] != v for k, v in hashes.items()):
                    raise ValueError(f'Changed checkpoint or config: {method}, {seed}')
                continue
            bundle = load_experiment(directory, seed)
            params = bundle.config['env']['params']
            matrix = np.array(params['transition_matrix']).reshape(3, 3)
            np.testing.assert_allclose(np.diag(matrix), .995, rtol=0, atol=1e-12)
            key_seed = 1_000_000 + 10_000 * index + seed
            rollout = collect_action_regime_rollouts(
                bundle.env, bundle.agent, bundle.agent_state, N_EPISODES,
                EPISODE_LENGTH, jax.random.PRNGKey(key_seed))
            row = summarize_rollout(rollout['regime'], rollout['reward'],
                                    rollout['post_inventory'], rollout['done'],
                                    params['inventory_penalty'])
            np.testing.assert_allclose(row['mean_episode_return'], reference[seed], rtol=0, atol=1e-10)
            row.update(method=method, seed=seed, evaluation_key=key_seed,
                       published_return=reference[seed], **hashes)
            payload['rows'].append(row)
            temporary = output.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
            temporary.replace(output)
            print(f"{method} seed {seed}: return verified; {row['switches']} switches, "
                  f"{row['episodes_without_switch']} no-switch episodes", flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
