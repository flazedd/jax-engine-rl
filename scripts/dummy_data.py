"""Deterministic schema/render fixtures, independent of untracked results.

The checked-in gzip fixture contains perturbed numerical examples with the
published artifact schemas. It exercises rendering, not training or statistical
correctness. All data are marked synthetic and only written to an isolated root.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from utils.paths import dummy_sibling, fig_targets, is_dummy, results_root

FIXTURE = Path(__file__).resolve().parents[1] / 'reproduction/fixtures/render_inputs.json.gz'


def fixtures():
    with gzip.open(FIXTURE, 'rt') as handle:
        return json.load(handle)


def synth_training_metrics(experiment):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(experiment.encode()).digest()[:4], 'little'))
    curve = np.linspace(5, 30, 1500)
    runs = curve + rng.normal(0, 1, (20, 1500))
    return {'dummy': True, 'experiment_name': experiment, 'iterations': 1500,
            'num_seeds': 20, 'seeds': list(range(20)), 'parallel_envs': 512, 'rollout_length': 128,
            'mean_return_per_iter': runs.mean(axis=0).tolist(),
            'var_return_per_iter': runs.var(axis=0).tolist(),
            'per_seed_mean_return_per_iter': runs.tolist(),
            'per_seed_final_return': runs[:,-1].tolist(), 'final_return_mean': float(runs[:,-1].mean()),
            'final_return_ci95': [28,32]}


def _environment_figures():
    from plotting.m2_plots import (plot_policy_heatmap, plot_value_loss_distribution,
                                   plot_per_regime_ppo, plot_belief_ppo_gap, plot_posterior_entropy)
    from envs.market_making_v1 import MarketMakingV1
    rng = np.random.default_rng(0)
    env = MarketMakingV1(inventory_max=5)
    calls = [
        ('fig_M2_R1_policy_heatmap.png', plot_policy_heatmap, (SimpleNamespace(Q=rng.normal(size=(11,3,3))), env)),
        ('fig_M2_R1_value_loss_distribution.png', plot_value_loss_distribution, (rng.uniform(0,0.8,60),)),
        ('fig_M2_R2_per_regime_ppo.png', plot_per_regime_ppo, ([synth_training_metrics(str(r)) for r in range(3)], np.array([35,40,45]))),
        ('fig_M2_R4_belief_ppo_gap.png', plot_belief_ppo_gap, ({'regime_agnostic':(20,[18,22]),'belief':(30,[28,32]),'oracle':(40,[38,42])},)),
        ('fig_M2_R4_posterior_entropy.png', plot_posterior_entropy, (np.linspace(1.09,0.2,128),)),
    ]
    for name, function, args in calls:
        for target in fig_targets(name):
            function(*args, target)


def write_stage(stage):
    if not is_dummy():
        raise RuntimeError('Synthetic fixture writer requires THESIS_DUMMY=1')
    examples = fixtures()
    for path in stage.outputs:
        if path.suffix != '.json':
            continue
        relative = str(path.relative_to(results_root()))
        if stage.kind == 'train':
            payload = (synth_training_metrics(path.parent.name) if path.name == 'metrics.json'
                       else {'dummy':True,'status':'OK','description':'Synthetic training marker'})
        else:
            if relative not in examples:
                raise ValueError(f'Missing versioned preflight schema: {relative}')
            payload = examples[relative]
        destination = dummy_sibling(path)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(payload))
    if stage.name == 'foundations:env_validation':
        _environment_figures()
    return True
