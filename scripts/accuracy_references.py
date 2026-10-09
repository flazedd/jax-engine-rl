"""Recover per-run reference curves for the paired accuracy and KL figures."""
import json
from pathlib import Path
import jax
import numpy as np
from evaluation.posterior_probe import load_experiment, collect_probe_rollouts, train_probe
from plotting.trading_results import _load_probe_for_per_t, MEDIUM_ENV, CELLS
from utils.paths import experiment_dir, analysis_dir


def main():
    output = analysis_dir() / 'm5r_accuracy_references.json'
    saved = json.loads(output.read_text()) if output.exists() else {'environment': MEDIUM_ENV, 'n_rollouts': 500, 'rollout_length': 128, 'methods': {}}
    sources = {c: _load_probe_for_per_t(c) for c in ('logistic', 'mlp')}
    for cell in CELLS:
        target = saved['methods'].setdefault(cell, {})
        seeds = [p['seed'] for p in sources['logistic']['scatter_points'] if p['method'] == cell and p['env_label'] == MEDIUM_ENV]
        for seed in seeds:
            if str(seed) in target and all(c + '_kl' in target[str(seed)] for c in sources):
                continue
            bundle = load_experiment(experiment_dir(f'm5r_final_{cell}_{MEDIUM_ENV}'), seed)
            data = collect_probe_rollouts(bundle.env, bundle.agent, bundle.agent_state, n_rollouts=500, rollout_length=128, key=jax.random.PRNGKey(seed))
            perm = np.random.default_rng(seed).permutation(500)
            result = {}
            for clf, source in sources.items():
                probe = train_probe(data['analytical_belief'], data['regime'], perm[100:], perm[:100], classifier=clf, seed=seed, omega_TND=data['analytical_belief'])
                original = next(p for p in source['scatter_points'] if p['method'] == cell and p['seed'] == seed and p['env_label'] == MEDIUM_ENV)
                if not np.isclose(probe['test_acc'], original['analytical_test_acc'], atol=1e-10):
                    raise ValueError(f'Replay mismatch {cell} {seed} {clf}: {probe["test_acc"]} vs {original["analytical_test_acc"]}')
                result[clf] = probe['per_t_test_acc']
                if not np.isclose(probe['test_kl_to_omega'], original['analytical_kl_to_omega'], rtol=1e-7, atol=1e-10):
                    raise ValueError(f'KL replay mismatch {cell} {seed} {clf}')
                result[clf + '_kl'] = probe['per_t_test_kl_to_omega']
            target[str(seed)] = result
            output.write_text(json.dumps(saved, indent=2) + '\n')
            print(f'{cell} seed {seed}: reference replay matches saved accuracy and KL', flush=True)


if __name__ == '__main__':
    main()
