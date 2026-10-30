"""Recalculate statistics and figures from the repository's pinned saved data."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import platform
import re
import shutil
from pathlib import Path
import subprocess
import sys

from scripts.prepare_results import DATA, digest, prepare

ROOT = Path(__file__).resolve().parents[1]
CONTROL_FILES = ('decision_loss_summary.json', 'decision_quality_comparisons.json',
                 'inventory_activity_summary.json')
CHAPTER_FIGURES = ('fig_rq1_learning_curves.png', 'method_ladder.png', 'learning_curves.png',
                   'switch_overview.pdf', 'return_components.pdf', 'inventory_activity.pdf',
                   'decision_quality.pdf')
SUPPORTING_TABLES = ('return_decomposition.tex', 'decision_quality_comparisons.tex',
                     'inventory_activity.tex', 'full_belief_probe.tex',
                     'natural_switch_counts.tex', 'natural_switch_differences.tex',
                     'decision_loss_switch_differences.tex')


def verify_control_inputs():
    manifests = {}
    for name in ('decision_loss_manifest.json', 'inventory_activity_manifest.json'):
        path = DATA / name
        manifest = json.loads(path.read_text())
        for relative, expected in manifest['sha256'].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f'Analysis checksum mismatch: {relative}')
        manifests[name] = digest(path)
    return manifests


def source_snapshot():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                                    cwd=ROOT, text=True).splitlines()
    allowed = {'.py', '.yaml', '.yml', '.toml', '.lock', '.md'}
    hashes = {name: digest(ROOT / name) for name in sorted(set(names))
              if (ROOT / name).is_file() and Path(name).suffix in allowed}
    return {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                     cwd=ROOT, text=True).strip(),
            'source_tree_dirty': bool(subprocess.check_output(
                ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'source_files_sha256': hashes}


def verify_outputs(output, thesis_root=None):
    paths = [output / 'figures/results/RQ1' / name for name in CHAPTER_FIGURES]
    paths += [output / 'tables' / name for name in SUPPORTING_TABLES]
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing or empty thesis output: {path}')
    if thesis_root:
        for name in SUPPORTING_TABLES:
            generated = (output / 'tables' / name).read_text()
            thesis = (thesis_root / 'tables' / name).read_text()
            def normalise(text):
                # LaTeX spacing commands do not change a table's values or units.
                return re.sub(r'\s+', '', re.sub(r'\\[ ,;!]|~', '', text))
            if normalise(generated) != normalise(thesis):
                raise ValueError(f'Generated appendix table differs from thesis: {name}')
    return {str(path.relative_to(output)): digest(path) for path in paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced',
                        help='New output directory; existing nonempty directories are rejected')
    parser.add_argument('--thesis-root', type=Path,
                        help='Also compare current thesis tables with regenerated values')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'{output} is not empty; choose a new --output directory')
    output.mkdir(parents=True, exist_ok=True)
    manifests = verify_control_inputs()
    prepare(DATA, output / 'results')
    control = output / 'results/analysis/frozen_policy'
    control.mkdir(parents=True)
    for name in CONTROL_FILES:
        shutil.copy2(DATA / name, control / name)
    env = dict(os.environ, THESIS_RESULTS_ROOT=str(output / 'results'),
               THESIS_PROJECT_FIGS=str(output / 'figures'),
               THESIS_FIG_ROOT=str(output / 'figures'),
               MPLCONFIGDIR=str(output / '.matplotlib'), THESIS_DUMMY='0')
    steps = [
        ('scripts.restore_validation_baselines', []),
        ('scripts.seed_block_sensitivity', []),
        ('scripts.exploratory_baseline_seed_pairs', []),
        ('scripts.primary_seed_block_analysis', []),
        ('scripts.make_tables', []),
        ('scripts.summarize_full_belief',
         ['--mlp', str(DATA / 'm5r_full_belief_mlp.json'),
          '--table', str(output / 'tables/full_belief_probe.tex')]),
        ('plotting.trading_results', []),
        ('plotting.reference_levels', []),
        ('plotting.action_inventory_heatmap', []),
        ('plotting.m4_plots', []),
        ('scripts.plot_supplemental_belief_checks',
         ['--output', str(output / 'figures/appendix/direct_posterior_accuracy.png')]),
        ('scripts.summarize_natural_switches', ['--output-root', str(output), '--verify-saved']),
        ('scripts.summarize_decision_loss',
         ['--input', str(control / 'decision_loss_summary.json'), '--tables', str(output / 'tables'),
          '--figures', str(output / 'figures/results/RQ1')]),
        ('scripts.summarize_quote_quality',
         ['--input', str(control / 'decision_quality_comparisons.json'),
          '--table', str(output / 'tables/decision_quality_comparisons.tex')]),
        ('scripts.plot_control_explanations',
         ['--input', str(control / 'decision_loss_summary.json'),
          '--output', str(output / 'figures/results/RQ1')]),
        ('scripts.plot_inventory_activity',
         ['--input', str(control / 'inventory_activity_summary.json'),
          '--figure', str(output / 'figures/results/RQ1/inventory_activity.pdf'),
          '--table', str(output / 'tables/inventory_activity.tex')]),
        ('scripts.plot_switch_overview',
         ['--archive', str(DATA / 'saved-results.tar.gz'),
          '--decision-loss', str(control / 'decision_loss_summary.json'),
          '--output', str(output / 'figures/results/RQ1/switch_overview.pdf')]),
        ('scripts.thesis_contract', ['--strict'] +
         (['--thesis-root', str(args.thesis_root.resolve())] if args.thesis_root else [])),
    ]
    report = {'status': 'running', 'steps': [], 'output': str(output),
              'runtime': {'python': sys.version, 'platform': platform.platform(),
                          'packages': {name: version(name) for name in
                                       ('numpy', 'matplotlib', 'jax', 'scikit-learn')}},
              'data_manifest_sha256': digest(DATA / 'manifest.json'),
              'analysis_manifests_sha256': manifests,
              'control_inputs_sha256': {name: digest(DATA / name) for name in CONTROL_FILES},
              **source_snapshot()}
    report_path = output / 'verification.json'
    for module, arguments in steps:
        print(f'Running {module}', flush=True)
        log_path = output / (module.replace('.', '_') + '.log')
        with log_path.open('w') as log:
            process = subprocess.run([sys.executable, '-m', module, *arguments],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['steps'].append({'module': module, 'exit_code': process.returncode,
                                'log': log_path.name})
        if process.returncode:
            report['status'] = 'failed'
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'Failed: see {log_path}', file=sys.stderr)
            return process.returncode
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        report['outputs_sha256'] = verify_outputs(output, args.thesis_root)
    except (OSError, ValueError) as exc:
        report.update(status='failed', error=str(exc))
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(str(exc), file=sys.stderr)
        return 1
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved result reproduction passed. Figures, tables, logs, and report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
