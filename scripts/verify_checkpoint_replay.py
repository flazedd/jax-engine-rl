"""Replay frozen policies against saved fresh-evaluation returns without overwriting them."""
import argparse
import json
import jax
import numpy as np
from evaluation.action_distribution import collect_action_regime_rollouts
from evaluation.posterior_probe import load_experiment
from scripts.post_training_evaluation import METHODS, N_EPISODES, EPISODE_LENGTH
from utils.paths import analysis_dir, experiment_dir


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-seeds',action='store_true',help='Replay all 160 checkpoints; default: seed 0 for each of eight conditions')
    parser.add_argument('--atol',type=float,default=1e-10)
    args=parser.parse_args()
    saved=json.loads((analysis_dir()/'m5r_post_training_evaluation.json').read_text())['methods']
    rows=[]
    for index,(method,experiment) in enumerate(METHODS):
        reference=dict(zip(saved[method]['seeds'],saved[method]['per_seed_evaluation_return']))
        for seed in (range(20) if args.all_seeds else [0]):
            bundle=load_experiment(experiment_dir(experiment),seed)
            rollout=collect_action_regime_rollouts(bundle.env,bundle.agent,bundle.agent_state,
                     n_rollouts=N_EPISODES,rollout_length=EPISODE_LENGTH,
                     key=jax.random.PRNGKey(1_000_000+10_000*index+seed))
            value=float(np.asarray(rollout['reward'],dtype=float).sum(axis=0).mean())
            row={'method':method,'seed':seed,'episodes':N_EPISODES,'observed':value,
                 'archived':reference[seed],'difference':value-reference[seed],
                 'matches':bool(np.isclose(value,reference[seed],rtol=0,atol=args.atol))}
            rows.append(row)
            print(json.dumps(row),flush=True)
    output=analysis_dir()/'checkpoint_replay_verification.json'
    output.write_text(json.dumps({'all_seeds':args.all_seeds,'absolute_tolerance':args.atol,'results':rows},indent=2))
    return 0 if all(r['matches'] for r in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
