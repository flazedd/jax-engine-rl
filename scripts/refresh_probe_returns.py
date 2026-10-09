"""Join saved probes to fresh evaluation returns by experiment and training seed.

Use --migrate-legacy-seeds once for the archived evaluator whose checkpoint loop
sorted filenames lexically. This explicit migration preserves the original order;
new evaluation files record seed IDs directly. No policy or probe is retrained.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from utils.paths import analysis_dir, experiment_dir


def evaluation_returns_for_seeds(methods, method, experiment, seeds):
    result = methods[method]
    if result['experiment'] != experiment:
        raise ValueError(f'Evaluation/probe experiment mismatch: {experiment}')
    ids = result['seeds']
    values = result['per_seed_evaluation_return']
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError('Evaluation seed IDs must be unique and match return count')
    by_seed = dict(zip(ids, values))
    if len(seeds) != len(ids) or len(set(seeds)) != len(seeds) or set(seeds) != set(ids):
        raise ValueError('Evaluation and probe seed sets differ')
    return [by_seed[s] for s in seeds]


def refresh_probe(probe, methods):
    from scripts.posterior_probe import _decoupling_diagnostic
    points = probe['scatter_points']
    floor = methods['regime_agnostic_ppo']['evaluation_return_mean']
    ceiling = methods['belief_ppo']['evaluation_return_mean']
    grouped = {}
    for point in points:
        grouped.setdefault((point['method'], point['experiment_name']), []).append(point)
    for (method, experiment), group in grouped.items():
        values = evaluation_returns_for_seeds(methods, method, experiment, [p['seed'] for p in group])
        for p, value in zip(group, values):
            p['evaluation_return'] = value
            p['gap_closed'] = (value-floor)/(ceiling-floor)
        for env in {p['env_label'] for p in group}:
            probe['per_method_per_env'][env][method]['seeds'] = [p['seed'] for p in group if p['env_label']==env]
    def corr(ps, key):
        return float(np.corrcoef([p[key] for p in ps], [p['gap_closed'] for p in ps])[0,1])
    for suffix, key in [('', 'posterior_error'), ('_kl', 'belief_error_kl')]:
        probe['correlation_overall'+suffix] = corr(points,key)
        probe['correlation_per_method'+suffix] = {m:corr(g,key) for (m,_),g in grouped.items()}
        probe['decoupling_detected'+suffix] = _decoupling_diagnostic(points,error_key=key)
    probe['return_source'] = 'm5r_post_training_evaluation.json; joined by experiment and seed'
    probe['reference_means'] = {'regime_agnostic_ppo':floor, 'belief_ppo':ceiling}
    probe['analysis_version'] = 2
    return probe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate-legacy-seeds', action='store_true')
    args=parser.parse_args()
    root=analysis_dir()
    for path in root.glob('m5r_post_training_evaluation*.json'):
        data=json.loads(path.read_text())
        changed=False
        for result in data.get('methods',{}).values():
            if 'seeds' in result: continue
            if not args.migrate_legacy_seeds:
                raise ValueError(f'{path}: missing seeds; explicitly migrate legacy evaluation first')
            paths=sorted(experiment_dir(result['experiment']).glob('checkpoint_seed_*.pkl'))
            seeds=[int(p.stem.rsplit('_',1)[1]) for p in paths]
            if sorted(seeds)!=list(range(20)) or len(result['per_seed_evaluation_return'])!=20:
                raise ValueError('Legacy migration requires the complete archived 20-seed experiment')
            result['seeds']=seeds
            result['seed_metadata_provenance']='Recovered from archived evaluator lexical checkpoint filename ordering; values unchanged'
            changed=True
        if changed: path.write_text(json.dumps(data,indent=2)+'\n')
    methods=json.loads((root/'m5r_post_training_evaluation.json').read_text())['methods']
    refreshed = []
    for suffix in ['', '_mlp']:
        path=root/f'm5r_posterior_vs_performance{suffix}.json'
        probe=refresh_probe(json.loads(path.read_text()),methods)
        path.write_text(json.dumps(probe,indent=2)+'\n')
        summary_path = root/f'm5r_posterior_vs_performance{suffix}_run.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary['key_stats']['correlation_overall'] = probe['correlation_overall']
            summary['key_stats']['correlation_overall_kl'] = probe['correlation_overall_kl']
            summary['analysis_refreshed_at'] = datetime.now(timezone.utc).isoformat()
            summary['analysis_refresh_script'] = 'scripts.refresh_probe_returns'
            summary['note'] = 'Original timestamps describe probe fitting; return associations refreshed without refitting'
            summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        refreshed.append(path.name)
        print(path.name, probe['correlation_overall_kl'])
    (root/'m5r_refresh_probe_returns_run.json').write_text(json.dumps({
        'script':'refresh_probe_returns','status':'OK',
        'finished_at':datetime.now(timezone.utc).isoformat(),
        'legacy_seed_migration':args.migrate_legacy_seeds,
        'outputs':refreshed, 'probe_refitting':False,
        'join_keys':['experiment','seed'],
        'return_source':'m5r_post_training_evaluation.json'},indent=2)+'\n')

if __name__=='__main__': main()
