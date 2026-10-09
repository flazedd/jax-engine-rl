"""Replay saved probes, verify original recovery bins, and retain finer age bins.

Run with python -m scripts.recovery_audit. No policies are retrained.
The direct posterior and fitted reference use exactly the method's test steps.
"""
import json
import argparse
from pathlib import Path

import jax
import numpy as np

from evaluation.posterior_probe import collect_probe_rollouts, load_experiment, train_probe
from plotting.trading_results import CELLS, MEDIUM_ENV, _load_probe_for_per_t
from utils.paths import analysis_dir, experiment_dir

BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19),
        (20, 39), (40, 59), (60, 79), (80, 99), (100, 126)]


def ages_since_change(truth):
    changed = np.zeros(truth.shape, dtype=bool)
    changed[1:] = truth[1:] != truth[:-1]
    times = np.arange(len(truth))[:, None]
    latest = np.maximum.accumulate(np.where(changed, times, -1), axis=0)
    return np.where(latest >= 0, times - latest, -1)


def summarize(correct, age):
    counts, scores = [], []
    for lo, hi in BINS:
        mask = (age >= lo) & (age <= hi)
        counts.append(int(mask.sum()))
        scores.append(float(correct[mask].mean()) if mask.any() else None)
    return scores, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=CELLS)
    parser.add_argument('--merge', action='store_true', help='Combine four completed method outputs')
    args = parser.parse_args()
    default_path = analysis_dir() / 'm5r_recovery_audit.json'
    if args.merge:
        rows = []
        for cell in CELLS:
            part = json.loads((analysis_dir() / f'm5r_recovery_audit_{cell}.json').read_text())
            if sorted(r['seed'] for r in part['rows']) != list(range(20)):
                raise ValueError(f'Incomplete method output: {cell}')
            if part['bins'] != [list(b) for b in BINS] or part['environment'] != MEDIUM_ENV:
                raise ValueError(f'Incompatible method output: {cell}')
            rows.extend(part['rows'])
        part['rows'] = rows
        part['verification'] = 'All 160 probe/reference pairs reproduce original overall accuracy and all seven original recovery bins'
        default_path.write_text(json.dumps(part, indent=2, allow_nan=False) + '\n')
        print(default_path)
        return
    path = default_path if args.method is None else analysis_dir() / f'm5r_recovery_audit_{args.method}.json'
    resume = path if path.exists() else default_path
    payload = json.loads(resume.read_text()) if resume.exists() else {
        'environment': MEDIUM_ENV, 'bins': BINS, 'n_rollouts': 500,
        'test_episodes': 100, 'timing': 'before action; age zero is first decision in new regime',
        'rows': [],
    }
    if args.method:
        payload['rows'] = [r for r in payload['rows'] if r['method'] == args.method]
    sources = {c: _load_probe_for_per_t(c) for c in ('logistic', 'mlp')}
    for cell in CELLS:
        if args.method and cell != args.method:
            continue
        for seed in range(20):
            if any(r['method'] == cell and r['seed'] == seed for r in payload['rows']):
                continue
            bundle = load_experiment(experiment_dir(f'm5r_final_{cell}_{MEDIUM_ENV}'), seed)
            data = collect_probe_rollouts(bundle.env, bundle.agent, bundle.agent_state,
                                          n_rollouts=500, rollout_length=128,
                                          key=jax.random.PRNGKey(seed))
            split = np.random.default_rng(seed).permutation(500)
            test, train = split[:100], split[100:]
            truth = data['regime'][:, test]
            age = ages_since_change(truth)
            direct_pred = data['analytical_belief'][:, test].argmax(-1)
            direct = direct_pred == truth
            row = {'method': cell, 'seed': seed,
                   'direct_accuracy': float(direct.mean()),
                   'direct_since_change': summarize(direct, age)[0],
                   'counts': summarize(direct, age)[1], 'probes': {}}
            previous = np.concatenate([truth[:1], truth[:-1]], axis=0)
            for classifier, source in sources.items():
                block = source['per_method_per_env'][MEDIUM_ENV][cell]
                points = [p for p in source['scatter_points']
                          if p['method'] == cell and p['env_label'] == MEDIUM_ENV]
                index = next(i for i,p in enumerate(points) if p['seed'] == seed)
                pair = {}
                for role, inputs in [('method', data['belief']),
                                     ('analytical', data['analytical_belief'])]:
                    result = train_probe(inputs, data['regime'], train, test,
                                         classifier=classifier, seed=seed,
                                         omega_TND=data['analytical_belief'], return_predictions=True)
                    # Verify replay against the exact original plotted bins and overall scores.
                    np.testing.assert_allclose(result['test_acc_since_change'],
                        block[role + '_test_acc_since_change_per_seed'][index], atol=1e-10)
                    np.testing.assert_allclose(result['test_acc'],
                        points[index][role + '_test_acc'], atol=1e-10)
                    pred = result['test_predictions']
                    pair[role] = {'accuracy': result['test_acc'],
                                  'since_change': summarize(pred == truth, age)[0],
                                  'previous_regime_at_change': float((pred[age == 0] == previous[age == 0]).mean())}
                row['probes'][classifier] = pair
            row['direct_previous_regime_at_change'] = float((direct_pred[age == 0] == previous[age == 0]).mean())
            payload['rows'].append(row)
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
            print(f'{cell} seed {seed}: both probes match saved scores and recovery bins', flush=True)
    print(path, flush=True)


if __name__ == '__main__':
    main()
