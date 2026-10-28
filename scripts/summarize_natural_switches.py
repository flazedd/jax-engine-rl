"""Regenerate descriptive natural-switch statistics, figure, and appendix tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plotting.style import apply_style
from scripts.natural_switch_evaluation import BINS, CELLS
from scripts.prepare_results import digest
from utils.paths import analysis_dir, thesis_fig_dir

LABELS = ['0', '1--4', '5--9', '10--19', '20--39', '40--79', '80--126']
NAMES = {'rl2_concat': r'RL\textsuperscript{2} concat.',
         'rl2_hypernet': r'RL\textsuperscript{2} hyper.',
         'varibad_concat': 'VariBAD concat.', 'varibad_hypernet': 'VariBAD hyper.'}


def estimate(values, indices):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Every included training seed must have a finite bin mean')
    means = values[indices].mean(axis=1)
    return {'mean': float(values.mean()),
            'ci95': np.percentile(means, [2.5, 97.5]).tolist()}


def summarize(payload):
    rows = payload['rows']
    if sorted((r['method'], r['seed']) for r in rows) != sorted(
            (c, s) for c in CELLS for s in range(20)):
        raise ValueError('Expected exactly 20 distinct seeds in each of four conditions')
    if payload['identity']['bins'] != [list(b) for b in BINS]:
        raise ValueError('Unexpected switch intervals')
    indices = np.random.default_rng(20261028).integers(0, 20, size=(10000, 20))
    result = {'bootstrap_resamples': 10000, 'bootstrap_seed': 20261028,
              'intervals': 'Pointwise percentile 95%; descriptive, not multiplicity-adjusted',
              'bin_aggregation': 'Step-weighted within each seed; equal weight across 20 seeds',
              'difference_aggregation': 'Hypernetwork minus concatenation; bootstrap matching training-seed pairs',
              'expected_no_switch_fraction': .995 ** 127, 'methods': {}, 'differences': {}}
    arrays = {}
    for cell in CELLS:
        runs = sorted((r for r in rows if r['method'] == cell), key=lambda r:r['seed'])
        for r in runs:
            if r['episodes'] != 512 or len(r['bins']) != len(BINS):
                raise ValueError('Unexpected episode or bin count')
            if not np.isclose(r['mean_episode_return'], r['published_return'], rtol=0, atol=1e-10):
                raise ValueError('Replay does not match published return')
            if [b['interval'] for b in r['bins']] != [list(b) for b in BINS]:
                raise ValueError('Run bin definitions differ')
        block = {k: sum(r[k] for r in runs) for k in (
            'episodes', 'switches', 'episodes_without_switch',
            'episodes_with_one_switch', 'episodes_with_multiple_switches')}
        block['bins'] = []
        arrays[cell] = {}
        for metric in ('mean_reward', 'mean_inventory_penalty'):
            arrays[cell][metric] = np.array([[b[metric] for b in r['bins']] for r in runs], dtype=float)
        for i, interval in enumerate(BINS):
            b = {'interval': list(interval), 'seeds': 20}
            b.update({k: sum(r['bins'][i][k] for r in runs) for k in ('episodes', 'switches', 'steps')})
            if any(r['bins'][i]['steps'] <= 0 for r in runs):
                raise ValueError('Empty seed/bin; do not silently change the seed population')
            for metric in arrays[cell]:
                b[metric] = estimate(arrays[cell][metric][:, i], indices)
            block['bins'].append(b)
        result['methods'][cell] = block
    for method in ('rl2', 'varibad'):
        result['differences'][method] = []
        for i, interval in enumerate(BINS):
            b = {'interval': list(interval), 'seed_pairs': 20}
            for metric in arrays[CELLS[0]]:
                values = arrays[method+'_hypernet'][metric][:, i] - arrays[method+'_concat'][metric][:, i]
                b[metric] = estimate(values, indices)
            result['differences'][method].append(b)
    return result


def write_outputs(result, root):
    figures, tables = root / 'figures/appendix', root / 'tables'
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    counts = [r'\begin{tabular}{llrrrr}', r'\toprule',
              r'Agent & Steps since switch & Seeds & Episodes & Switches & Steps \\', r'\midrule']
    for cell in CELLS:
        for i, b in enumerate(result['methods'][cell]['bins']):
            counts.append(f"{NAMES[cell] if i == 0 else ''} & {LABELS[i]} & {b['seeds']} & "
                          f"{b['episodes']:,} & {b['switches']:,} & {b['steps']:,} " + r'\\')
        counts.append(r'\midrule' if cell != CELLS[-1] else r'\bottomrule')
    counts.append(r'\end{tabular}')
    (tables / 'natural_switch_counts.tex').write_text('\n'.join(counts)+'\n')
    differences = [r'\begin{tabular}{llrr}', r'\toprule',
                   r'Method & Steps since switch & Reward difference [95\% CI] & Penalty difference [95\% CI] \\', r'\midrule']
    for method, name in [('rl2', r'RL\textsuperscript{2}'), ('varibad', 'VariBAD')]:
        for i, b in enumerate(result['differences'][method]):
            cells = []
            for metric in ('mean_reward', 'mean_inventory_penalty'):
                e = b[metric]; lo, hi = e['ci95']
                cells.append(f"${e['mean']:+.4f}\\ [{lo:+.4f},{hi:+.4f}]$")
            differences.append(f"{name if i == 0 else ''} & {LABELS[i]} & " + ' & '.join(cells) + r' \\')
        differences.append(r'\midrule' if method == 'rl2' else r'\bottomrule')
    differences.append(r'\end{tabular}')
    (tables / 'natural_switch_differences.tex').write_text('\n'.join(differences)+'\n')
    apply_style()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
    colours = ['#6c7a80', '#177e78', '#a6b2bb', '#73c7bd']
    names = ['RL² concat.', 'RL² hypernetwork', 'VariBAD concat.', 'VariBAD hypernetwork']
    x = np.arange(len(BINS))
    for j, metric in enumerate(('mean_reward', 'mean_inventory_penalty')):
        for cell, colour, name in zip(CELLS, colours, names):
            estimates = [b[metric] for b in result['methods'][cell]['bins']]
            y = np.array([e['mean'] for e in estimates]); ci = np.array([e['ci95'] for e in estimates])
            axes[0, j].plot(x, y, color=colour, marker='o', ms=3,
                            linestyle='--' if cell.startswith('varibad') else '-', label=name)
            axes[0, j].fill_between(x, ci[:, 0], ci[:, 1], color=colour, alpha=.13)
        for method, colour, name in [('rl2', '#177e78', 'RL²'), ('varibad', '#94603c', 'VariBAD')]:
            estimates = [b[metric] for b in result['differences'][method]]
            y = np.array([e['mean'] for e in estimates]); ci = np.array([e['ci95'] for e in estimates])
            axes[1, j].plot(x, y, color=colour, marker='o', ms=3, label=name)
            axes[1, j].fill_between(x, ci[:, 0], ci[:, 1], color=colour, alpha=.15)
        axes[1, j].axhline(0, color='#777777', linewidth=.8, linestyle=':')
        axes[1, j].set_xticks(x, [s.replace('--', '–') for s in LABELS], rotation=25)
        axes[1, j].set_xlabel('Steps since most recent regime switch (interval)')
    axes[0, 0].set_ylabel('Mean reward per step'); axes[0, 1].set_ylabel('Mean inventory penalty per step')
    axes[1, 0].set_ylabel('Reward: hypernetwork − concat.')
    axes[1, 1].set_ylabel('Penalty: hypernetwork − concat.')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8, ncol=2); axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / 'natural_switch_performance.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=analysis_dir() / 'natural_switch_evaluation.json')
    parser.add_argument('--output-root', type=Path, default=thesis_fig_dir().parent)
    parser.add_argument('--verify-saved', action='store_true',
                        help='Require recomputation to match the packaged summary before writing')
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    root = Path(__file__).resolve().parents[1]
    for relative, expected in payload['identity']['source_sha256'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'Evaluation source checksum mismatch: {relative}')
    reference = analysis_dir() / 'm5r_post_training_evaluation.json'
    if digest(reference) != payload['identity']['evaluation_reference_sha256']:
        raise ValueError('Published evaluation reference checksum mismatch')
    saved = json.loads(reference.read_text())['methods']
    for row in payload['rows']:
        block = saved[row['method']]
        expected = dict(zip(block['seeds'], block['per_seed_evaluation_return']))[row['seed']]
        if row['published_return'] != expected:
            raise ValueError('Run reference differs from published evaluation')
    result = summarize(payload)
    result['input_sha256'] = digest(args.input)
    path = analysis_dir() / 'natural_switch_summary.json'
    if args.verify_saved and (not path.exists() or json.loads(path.read_text()) != result):
        raise ValueError('Packaged natural-switch summary does not match recomputation')
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_outputs(result, args.output_root)
    print(path)


if __name__ == '__main__':
    main()
