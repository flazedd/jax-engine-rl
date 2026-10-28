"""Publish seed-block inference while retaining the original independence analysis.

No models are trained and no fitted measurements change. The six original
comparison families remain fixed; full-input probes remain exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.seed_block_sensitivity import families_from_artifacts, paired_comparison
from utils.paths import analysis_dir, resolve_data


FILENAME = 'm5r_primary_seed_block_analysis.json'


def load_primary(root: Path | None = None) -> dict:
    folder = root / 'analysis' if root is not None else analysis_dir()
    return json.loads(resolve_data(folder / FILENAME).read_text())


def main() -> int:
    block = json.loads((analysis_dir() / 'm5r_seed_block_sensitivity.json').read_text())
    paired = {(r['family'], r['comparison'], r['metric']): r for r in block['comparisons']}
    families = {}
    for family, entries in families_from_artifacts().items():
        rows = []
        for original, a, b in entries:
            key = (family, original.get('name', original.get('method', '')), original.get('metric', ''))
            result = paired[key]
            if not np.isclose(np.mean(a-b), result['delta'], atol=1e-12, rtol=0):
                raise ValueError(f'Seed-block mean mismatch: {key}')
            row = {k: v for k, v in original.items()
                   if k not in {'wilcoxon_p', 'wilcoxon_null_distribution', 'rank_biserial',
                                'stable_under_seed_omission', 'stable_under_run_omission'}}
            row.update(mean_difference=result['delta'], mean_paired_delta=result['delta'],
                       delta_ci=result['paired_bootstrap_ci'],
                       permutation_p=result['paired_signflip_p'],
                       holm_corrected_p=result['paired_p_holm'],
                       supported=result['paired_p_holm'] < .05,
                       ci_excludes_zero=bool(np.prod(result['paired_bootstrap_ci']) > 0),
                       inference='paired bootstrap and paired sign-flip test',
                       independence_sensitivity={k: original[k] for k in
                            ('mean_difference', 'delta_ci', 'permutation_p', 'holm_corrected_p')})
            if family == 'architecture_returns':
                row['seeds_favouring_hypernet'] = int(np.count_nonzero(a > b))
                row['n_pairs'] = len(a)
                row['hypernet_std'] = float(np.std(a, ddof=1))
                row['concat_std'] = float(np.std(b, ddof=1))
            rows.append(row)
        families[family] = rows

    # Proper-score checks remain exploratory; give them the same pairing as
    # the main mean-only probe comparisons, without adding a correction family.
    robustness = []
    for classifier, suffix in [('linear', ''), ('mlp', '_mlp')]:
        data = json.loads((analysis_dir() / f'm5r_posterior_vs_performance{suffix}.json').read_text())
        lookup = {(p['method'], p['seed']): p for p in data['scatter_points']}
        for method in ('rl2', 'varibad'):
            for metric in ('method_log_loss', 'method_brier'):
                a, b = [np.array([lookup[method+'_'+arch, s][metric] for s in range(20)])
                        for arch in ('hypernet', 'concat')]
                difference, interval, _ = paired_comparison(a, b, n_permutations=1)
                robustness.append(dict(method=method, probe=classifier, metric=metric,
                    hypernet_mean=float(a.mean()), concat_mean=float(b.mean()),
                    mean_difference=difference, mean_paired_delta=difference,
                    delta_ci=interval, seeds_favouring_hypernet=int(np.count_nonzero(a < b)),
                    n_pairs=20, inference='exploratory paired bootstrap; no hypothesis test'))
    output = dict(analysis_status='primary seed-block inference, revised after review; not preregistered',
                  unchanged=['checkpoints', 'evaluation returns', 'fitted probe measurements',
                             'six comparison families', 'full-input probe exploratory status'],
                  assumption=block['assumption'], families=families,
                  robustness_comparisons=robustness, correlations=block['correlations'],
                  independence_sensitivity_decisions_changed=sum(r['significance_changed'] for r in paired.values()))
    path = analysis_dir() / FILENAME
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'Published {sum(map(len, families.values()))} primary seed-block comparisons: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
