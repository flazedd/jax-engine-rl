"""Replay frozen policies to measure post-fill inventory and accepted bid/ask fills."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHOD_NAMES = ('rl2_concat', 'rl2_hypernet', 'varibad_concat', 'varibad_hypernet')


def accepted_fills(action, pre, post, reward, penalty):
    """Recover both accepted flags, including simultaneous fills, from exact accounting.

    dq=bid-ask and gain=bid_distance*bid+ask_distance*ask uniquely determine both.
    Post inventory is recorded before terminal reset. Raw attempted fills are not counted.
    """
    from envs.market_making_v1 import _BID_SPREADS, _ASK_SPREADS
    bid_distance = np.asarray(_BID_SPREADS)[action]
    ask_distance = np.asarray(_ASK_SPREADS)[action]
    gain = np.asarray(reward, dtype=float) + penalty * np.asarray(post, dtype=float)**2
    dq = np.asarray(post) - np.asarray(pre)
    bid = (gain + ask_distance*dq)/(bid_distance+ask_distance)
    ask = bid-dq
    np.testing.assert_allclose(bid, np.rint(bid), atol=2e-6, rtol=0)
    np.testing.assert_allclose(ask, np.rint(ask), atol=2e-6, rtol=0)
    bid, ask = np.rint(bid).astype(int), np.rint(ask).astype(int)
    assert np.isin(bid, [0, 1]).all() and np.isin(ask, [0, 1]).all()
    np.testing.assert_allclose(post, pre+bid-ask, atol=0)
    np.testing.assert_allclose(reward, bid_distance*bid+ask_distance*ask-penalty*post**2,
                               atol=2e-6, rtol=0)
    return bid, ask


def interval(values):
    values=np.asarray(values, dtype=float)
    ix=np.random.default_rng(0).integers(len(values), size=(10000,len(values)))
    return {'mean':values.mean(axis=0).tolist(),
            'ci95':np.percentile(values[ix].mean(axis=1),[2.5,97.5],axis=0).tolist()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import jax
    from evaluation.action_distribution import collect_action_regime_rollouts
    from evaluation.posterior_probe import load_experiment
    from scripts.post_training_evaluation import METHODS, N_EPISODES, EPISODE_LENGTH
    from utils.paths import experiment_dir
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/analysis/inventory_activity')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(20)))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    expected=json.loads((ROOT/'reproduction/data/decision_loss_summary.json').read_text())
    reference={(r['method'],r['seed']):r for r in expected['rows']}
    rows=[]
    for index,(method,exp) in enumerate(METHODS):
        if method not in METHOD_NAMES:continue
        directory=experiment_dir(exp)
        for seed in args.seeds:
            ck=directory/f'checkpoint_seed_{seed}.pkl'
            identity={'checkpoint':sha(ck),'config':sha(directory/'config.json'),
                      'collector':sha(ROOT/'evaluation/action_distribution.py'),
                      'analysis':sha(__file__)}
            dest=args.output/f'{method}_{seed}.json'
            if dest.exists():
                row=json.loads(dest.read_text());assert row['identity']==identity
            else:
                bundle=load_experiment(directory,seed)
                penalty=bundle.config['env']['params']['inventory_penalty']
                limit=bundle.config['env']['params']['inventory_max']
                key=1000000+10000*index+seed
                roll=collect_action_regime_rollouts(bundle.env,bundle.agent,bundle.agent_state,
                                                    N_EPISODES,EPISODE_LENGTH,jax.random.PRNGKey(key))
                pre,post=roll['inventory'],roll['post_inventory'];reward=roll['reward'].astype(float)
                bid,ask=accepted_fills(roll['action'],pre,post,reward,penalty)
                # Blocked sides are rejected using pre-fill inventory, including when both raw draws fill.
                assert (bid[pre==limit]==0).all() and (ask[pre==-limit]==0).all()
                ret=float(reward.sum(0).mean());ref=reference[method,seed]
                assert identity['checkpoint']==ref['identity']['checkpoint']
                np.testing.assert_allclose(ret,ref['return'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(penalty*(post.astype(float)**2).sum(0).mean(),ref['penalty'],atol=1e-10,rtol=0)
                row={'method':method,'seed':seed,'identity':identity,'evaluation_key':key,
                     'episodes':N_EPISODES,'steps_per_episode':EPISODE_LENGTH,'return':ret,
                     'absolute_inventory_distribution':(np.bincount(np.abs(post).ravel(),minlength=limit+1)/post.size).tolist(),
                     'mean_abs_inventory':float(np.abs(post).mean()),
                     'mean_squared_inventory':float((post.astype(float)**2).mean()),
                     'large_inventory_fraction':float((np.abs(post)>=3).mean()),
                     'bound_fraction':float((np.abs(post)==limit).mean()),
                     'bid_fills':float(bid.sum(0).mean()),'ask_fills':float(ask.sum(0).mean()),
                     'total_fills':float((bid+ask).sum(0).mean()),
                     'both_fill_steps':float(((bid==1)&(ask==1)).sum(0).mean())}
                dest.write_text(json.dumps(row,indent=2)+'\n')
            rows.append(row);print(method,seed,'verified',flush=True)
    metrics=['absolute_inventory_distribution','mean_abs_inventory','mean_squared_inventory',
             'large_inventory_fraction','bound_fraction','bid_fills','ask_fills','total_fills','both_fill_steps']
    summary={'description':'Post-fill absolute inventory and accepted fills on original evaluation episodes; no retraining.',
             'inventory_timing':'After accepted fills, before terminal reset; includes final decision.',
             'intervals':'10000 seed bootstrap resamples; RNG 0; paired differences; runs equally weighted.',
             'rows':rows,'methods':{},'comparisons':{}}
    for method in METHOD_NAMES:
        rr=[r for r in rows if r['method']==method]
        summary['methods'][method]={m:interval([r[m] for r in rr]) for m in metrics}
    pairs=[('rl2','rl2_hypernet','rl2_concat'),('varibad','varibad_hypernet','varibad_concat'),
           ('concat_method','rl2_concat','varibad_concat'),('hypernet_method','rl2_hypernet','varibad_hypernet')]
    lookup={(r['method'],r['seed']):r for r in rows}
    for name,first,second in pairs:
        summary['comparisons'][name]={'first':first,'second':second,
            **{m:interval([np.asarray(lookup[first,s][m])-np.asarray(lookup[second,s][m]) for s in args.seeds]) for m in metrics}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
