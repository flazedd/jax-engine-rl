"""Recompute published mean comparisons and post-audit common-seed sensitivity.

The working-independence analysis is retained for transparency. Paired bootstrap
and sign flips respect shared training-seed blocks. The latter assumes symmetric
paired differences / within-pair exchangeability under the null, not guaranteed
by shared PRNG keys alone. Six Holm families are unchanged.
"""
from pathlib import Path
import json
import numpy as np
from evaluation.metrics import bootstrap_independent_mean_ci, permutation_mean_test, holm_bonferroni
from evaluation.action_distribution import regime_separation_per_seed
from scripts.m5r_return_tests import _first_reach
from utils.paths import analysis_dir, results_root, thesis_fig_dir


def read(name):
    return json.loads((analysis_dir()/f'{name}.json').read_text())


def aligned(seeds, values):
    if len(seeds) != 20 or sorted(seeds) != list(range(20)) or len(values) != 20:
        raise ValueError('Expected one observation for every seed 0..19')
    return np.asarray([dict(zip(seeds, values))[s] for s in sorted(range(20), key=lambda i: f"checkpoint_seed_{i}.pkl")], dtype=float)


def paired_comparison(a, b, n_boot=10000, n_permutations=100000):
    d = np.asarray(a) - np.asarray(b)
    if d.shape != (20,) or not np.isfinite(d).all():
        raise ValueError('Expected 20 finite paired differences')
    rng = np.random.default_rng(0)
    boot = d[rng.integers(len(d), size=(n_boot, len(d)))].mean(axis=1)
    rng = np.random.default_rng(0)
    signs = rng.choice([-1, 1], size=(n_permutations, len(d)))
    p = (1 + np.count_nonzero(np.abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-12))/(n_permutations+1)
    return float(d.mean()), np.percentile(boot, [2.5,97.5]).tolist(), float(p)


def families_from_artifacts():
    ev=read('m5r_post_training_evaluation')['methods']
    ret=lambda m: aligned(ev[m]['seeds'], ev[m]['per_seed_evaluation_return'])
    families={}
    families['architecture_returns']=[(r,ret(r['method']),ret(r['baseline'])) for r in read('m5r_hypothesis_tests')['family_b']['results'].values()]
    families['method_returns']=[(r,ret('rl2_'+r['architecture']),ret('varibad_'+r['architecture'])) for r in read('m5r_method_return_tests')['comparisons']]
    families['speed']=[]
    for r in read('m5r_time_to_threshold_tests')['comparisons']:
        arrays=[]
        for arch in ['hypernet','concat']:
            p=results_root()/'medium'/f'm5r_final_{r["method"]}_{arch}_e9'/'metrics.json'
            curves=json.loads(p.read_text())['per_seed_mean_return_per_iter']
            reaches=[_first_reach(np.asarray(c),r['target_return']) for c in curves]
            arrays.append(np.asarray([1500 if x is None else x for x in reaches]))
        families['speed'].append((r,*arrays))
    for suffix in ['', '_mlp']:
        pts=read('m5r_posterior_vs_performance'+suffix)['scatter_points']
        comps=read('m5r_belief_quality'+('_mlp' if suffix else '')+'_tests')['comparisons']
        families['probe'+suffix]=[]
        for r in comps:
            arrays=[]
            for arch in ['hypernet','concat']:
                rows=sorted([p for p in pts if p['method']==r['method']+'_'+arch],key=lambda p:p['seed'])
                arrays.append(np.asarray([p[r['metric']] for p in rows]))
            families['probe'+suffix].append((r,*arrays))
    action=read('m5r_action_distributions')['by_method']
    swaps=read('m5r_belief_swap')['by_method']
    diag=read('m5r_diagnostic_tests')['results']
    families['diagnostics']=[]
    for tag in ['locked_regime_action_distribution','belief_swap_belief_only']:
        for method,label in [('rl2','RL2'),('varibad','VariBAD')]:
            arrays=[]
            for arch in ['hypernet','concat']:
                name=method+'_'+arch
                if tag.startswith('locked'):
                    b=action[name]
                    arr=aligned(b['seeds'], regime_separation_per_seed(b['per_seed_action_given_regime_inventory'],b['per_seed_inventory_counts']))
                else: arr=aligned(swaps[name]['seeds'], swaps[name]['per_seed_separation'])
                arrays.append(np.asarray(arr))
            families['diagnostics'].append((diag[tag][label],*arrays))
    return families


def correlation_sensitivity(points):
    methods = sorted({p['method'] for p in points})
    lookup = {(p['method'], p['seed']): p for p in points}
    if len(points) != 80 or len(lookup) != 80:
        raise ValueError('Expected four methods by twenty unique seeds')
    x = np.array([[lookup[m, s]['belief_error_kl'] for s in range(20)] for m in methods])
    y = np.array([[lookup[m, s]['gap_closed'] for s in range(20)] for m in methods])
    def corr(a,b,center):
        if center:
            a=a-a.mean(axis=-1,keepdims=True); b=b-b.mean(axis=-1,keepdims=True)
        return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    indices = np.random.default_rng(0).integers(20,size=(10000,20))
    return {k: {'r': corr(x,y,c), 'seed_block_ci': np.percentile(
        [corr(x[:,ix],y[:,ix],c) for ix in indices],[2.5,97.5]).tolist()}
        for k,c in [('overall',False),('within_variant',True)]}


def main():
    rows=[]
    for family, entries in families_from_artifacts().items():
        group=[]; originals=[]
        for saved,a,b in entries:
            mean,lo,hi=bootstrap_independent_mean_ci(a,b)
            p=permutation_mean_test(a,b)['p']
            if not np.allclose([mean,lo,hi,p], [saved['mean_difference'],*saved['delta_ci'],saved['permutation_p']], rtol=0,atol=1e-12):
                raise ValueError(f'Published comparison does not reproduce: {family}')
            originals.append(p)
            delta,ci,paired_p=paired_comparison(a,b)
            group.append({'family':family, 'comparison':saved.get('name', saved.get('method', '')),
                          'metric':saved.get('metric',''), 'delta':delta,
                          'paired_bootstrap_ci':ci, 'paired_signflip_p':paired_p,
                          'original_p_holm':saved['holm_corrected_p']})
        for row,adj,original_adj in zip(group,holm_bonferroni([r['paired_signflip_p'] for r in group]),holm_bonferroni(originals)):
            if not np.isclose(row['original_p_holm'],original_adj,rtol=0,atol=1e-12):
                raise ValueError('Original Holm adjustment does not reproduce')
            row['paired_p_holm']=adj
            row['significance_changed']=bool((adj<.05)!=(original_adj<.05))
        rows.extend(group)
    out={'analysis_status':'post-audit sensitivity; working-independence estimates retained',
         'assumption':'Paired sign flips require symmetric differences / within-pair exchangeability under the null.',
         'seed_ids':list(range(20)), 'bootstrap_resamples':10000, 'sign_flips':100000,
         'comparisons':rows, 'correlations':{label:correlation_sensitivity(read('m5r_posterior_vs_performance'+suffix)['scatter_points'])
                                          for suffix,label in [('', 'linear'),('_mlp','mlp')]}}
    path=analysis_dir()/'m5r_seed_block_sensitivity.json'
    path.write_text(json.dumps(out,indent=2))
    print(f'Recomputed {len(rows)} comparisons; {sum(r["significance_changed"] for r in rows)} decisions changed. Wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
