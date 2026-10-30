"""Verify the saved common-state estimates and render the appendix comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ('rl2_concat', 'rl2_hypernet', 'varibad_concat', 'varibad_hypernet')
PAIRS = ((METHODS[1], METHODS[0]), (METHODS[3], METHODS[2]),
         (METHODS[0], METHODS[2]), (METHODS[1], METHODS[3]))
LABELS = (r'RL\textsuperscript{2}: hypernet vs concat.', 'VariBAD: hypernet vs concat.',
          r'Concat.: RL\textsuperscript{2} vs VariBAD', r'Hypernet: RL\textsuperscript{2} vs VariBAD')


def verify_common_states(payload):
    common = payload['common_states']
    rows = sorted(common['rows'], key=lambda r: r['seed'])
    if [r['seed'] for r in rows] != list(range(20)):
        raise ValueError('Common-state summary must contain exactly seeds 0 through 19')
    levels = np.array([[r['loss'][m] for m in METHODS] for r in rows])
    coverage = np.array([[r['coverage'][m] for m in METHODS] for r in rows])
    if (not np.isfinite(levels).all() or (levels < 0).any()
            or not np.isfinite(coverage).all() or (coverage <= 0).any() or (coverage > 1).any()):
        raise ValueError('Invalid decision losses or coverage')
    np.testing.assert_allclose(coverage, np.repeat(coverage[:, :1], 4, axis=1), rtol=0, atol=1e-12)
    indices = np.random.default_rng(0).integers(20, size=(10_000, 20))

    def check(values, saved):
        np.testing.assert_allclose(saved['mean'], values.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(saved['ci95'], np.percentile(values[indices].mean(1), [2.5, 97.5]),
                                   rtol=0, atol=1e-12)

    for i, method in enumerate(METHODS):
        check(levels[:, i], common['configurations'][method])
        check(coverage[:, i], common['coverage'][method])
        np.testing.assert_allclose([common['coverage'][method]['min'], common['coverage'][method]['max']],
                                   [coverage[:, i].min(), coverage[:, i].max()], rtol=0, atol=1e-12)
    if [(c['first'], c['second']) for c in common['comparisons']] != list(PAIRS):
        raise ValueError('Unexpected common-state comparison order')
    if [(c['first'], c['second']) for c in payload['comparisons']] != list(PAIRS):
        raise ValueError('Unexpected supporting comparison order')
    for (first, second), comparison in zip(PAIRS, common['comparisons']):
        check(levels[:, METHODS.index(second)] - levels[:, METHODS.index(first)], comparison['reduction'])
    return {'runs_per_configuration': 20, 'mean_coverage': float(coverage[:, 0].mean()),
            'coverage_range': [float(coverage[:, 0].min()), float(coverage[:, 0].max())]}


def write_table(payload, path):
    lines = [r'\begin{tabular}{llrr}', r'\toprule',
             r'Comparison & Decisions & Reduction & $95\%$ interval \\', r'\midrule']
    for label, comparison, common in zip(LABELS, payload['comparisons'], payload['common_states']['comparisons']):
        for kind, estimate in [('All recorded', comparison['all_decisions']),
                               ('Comparable', comparison['comparable_situations']),
                               ('Common four', common['reduction'])]:
            lo, hi = estimate['ci95']
            lines.append(f"{label} & {kind} & ${estimate['mean']:.4f}$ & $[{lo:.4f}, {hi:.4f}]$" + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    report = verify_common_states(payload)
    write_table(payload, args.table)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
