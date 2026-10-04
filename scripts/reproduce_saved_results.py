"""Recalculate statistics and figures from the repository's pinned saved data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.prepare_results import DATA, prepare

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced',
                        help='New output directory; existing nonempty directories are rejected')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'{output} is not empty; choose a new --output directory')
    output.mkdir(parents=True, exist_ok=True)
    prepare(DATA, output / 'results')
    env = dict(os.environ, THESIS_RESULTS_ROOT=str(output / 'results'),
               THESIS_PROJECT_FIGS=str(output / 'figures'),
               THESIS_FIG_ROOT=str(output / 'figures'),
               MPLCONFIGDIR=str(output / '.matplotlib'), THESIS_DUMMY='0')
    steps = [
        ('scripts.restore_validation_baselines', []),
        ('scripts.m5r_seed_block_sensitivity', []),
        ('scripts.m5r_exploratory_baseline_seed_pairs', []),
        ('scripts.make_tables', []),
        ('plotting.m5r_plots', []),
        ('plotting.reference_levels', []),
        ('plotting.m5r_action_inventory_heatmap', []),
        ('plotting.m4_plots', []),
        ('scripts.plot_m5r_supplemental_belief_checks',
         ['--output', str(output / 'figures/appendix/m5r_direct_posterior_accuracy.png')]),
        ('scripts.thesis_contract', ['--strict']),
    ]
    report = {'status': 'running', 'steps': [], 'output': str(output)}
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
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved result reproduction passed. Figures, tables, logs, and report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
