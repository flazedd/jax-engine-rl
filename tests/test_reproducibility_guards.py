"""Regression tests for failures demonstrated by the October thesis audit."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import run_matched_programme as driver
from scripts.thesis_contract import _check_comparisons, _required_inputs
from training.config import AgentConfig, EnvConfig, ExperimentConfig
from utils import training_identity as identity


def test_parameter_audit_uses_main_programme_configurations():
    from scripts.param_counts import TARGETS
    from training.config import CONFIG_ROOT, load_config

    audited = {load_config(CONFIG_ROOT / relative).experiment_name
               for _, relative in TARGETS}
    scheduled = {load_config(path).experiment_name
                 for path in driver.CONFIGS.glob('*.yaml')}
    assert len(audited) == 8
    assert audited == scheduled


def configuration():
    return ExperimentConfig('test_identity', EnvConfig('dummy'), AgentConfig('dummy'),
                            iterations=2, num_seeds=1, parallel_envs=16, rollout_length=32).to_dict()


def seed_cache(folder, config):
    expected = identity.identity(config)
    folder.mkdir(exist_ok=True)
    (folder/'config.json').write_text(json.dumps(config))
    (folder/'provenance.json').write_text(json.dumps(expected))
    checkpoint = folder/'checkpoint_seed_0.pkl'
    checkpoint.write_bytes(b'regression fixture: identity check does not unpickle')
    cache = {'seed':0, 'mean_return_per_iter':[1,2], 'var_return_per_iter':[1,1],
             'per_iter_times':[1,1], '_training_fingerprint':expected['fingerprint'],
             '_checkpoint_sha256':identity.sha256(checkpoint)}
    (folder/'seed_0_result.json').write_text(json.dumps(cache))
    return expected


@pytest.mark.parametrize('change', ['environment', 'architecture', 'budget'])
def test_changed_training_identity_never_relabels_existing_run(tmp_path, change):
    config = configuration()
    seed_cache(tmp_path, config)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    if change == 'environment':
        config['env']['params']['kappa'] = .12345
    elif change == 'architecture':
        config['agent']['params']['hidden_size'] = 128
    else:
        config['iterations'] = 1500
    with pytest.raises(ValueError, match='Refusing to alter'):
        identity.prepare(tmp_path, identity.identity(config))
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()} == before


def test_checkpoint_corruption_and_incomplete_curves_rejected(tmp_path):
    expected = seed_cache(tmp_path, configuration())
    assert identity.validate_seed(tmp_path, 0, expected)['seed'] == 0
    (tmp_path/'checkpoint_seed_0.pkl').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        identity.validate_seed(tmp_path, 0, expected)
    expected = seed_cache(tmp_path, configuration())
    path=tmp_path/'seed_0_result.json';data=json.loads(path.read_text())
    data['mean_return_per_iter']=[1];path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='incomplete'):
        identity.validate_seed(tmp_path, 0, expected)


def test_empty_checkpoints_are_not_training_completion(tmp_path):
    for seed in range(20):
        (tmp_path/f'checkpoint_seed_{seed}.pkl').touch()
    (tmp_path/'summary.json').write_text('{"status":"OK"}')
    (tmp_path/'provenance.json').write_text('{}')
    assert not driver._training_is_complete(tmp_path/'summary.json')


def test_nested_architecture_family_is_audited():
    comparison = {'mean_difference':1, 'delta_ci':[.5,1.5], 'permutation_p':.01,
                  'holm_corrected_p':.02, 'stable_under_run_omission':True}
    findings=[]
    count=_check_comparisons({'family_b':{'results':{'rl2':comparison,'varibad':comparison}}}, 'returns_rsmm', findings)
    assert count == 2 and findings == []
    findings=[]
    assert _check_comparisons({}, 'returns_rsmm', findings) == 0
    assert findings[0]['kind'] == 'comparison_count'


def test_nonfinite_comparison_fails():
    findings=[]
    _check_comparisons({'comparisons':[{'mean_difference':float('nan'), 'delta_ci':[1,0],
                        'permutation_p':2,'holm_corrected_p':.2,'stable_under_run_omission':True}]}, 'returns_method', findings)
    assert sum(f['kind']=='invalid_comparison_value' for f in findings) == 3


def test_required_inputs_do_not_silently_skip_missing_files(tmp_path):
    findings=_required_inputs(tmp_path)
    missing={f['path'] for f in findings if f['kind']=='missing_or_corrupt_artifact'}
    assert {'analysis/m5r_post_training_evaluation.json',
            'analysis/m5r_posterior_vs_performance_mlp.json',
            'analysis/m5r_belief_swap.json'} <= missing


def test_graph_has_all_scientific_and_secondary_dependencies():
    stages={s.name:s for s in driver.build_plan()}
    assert {'foundations:env_validation','foundations:impl_validation'} <= set(stages['train:belief_ppo'].depends_on)
    assert 'analysis:posterior_probe_mlp' in stages['analysis:belief_quality_tests'].depends_on
    assert 'analysis:probe_confusion' in stages['figures:tables'].depends_on
    assert 'analysis:diagnostic_tests' in stages['figures:main'].depends_on
    assert {p.name for p in stages['analysis:return_tests'].outputs} == {'m5r_method_return_tests.json','m5r_time_to_threshold_tests.json'}
    assert len(stages['figures:tables'].outputs) == 7


def test_missing_secondary_output_invalidates_cached_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(driver,'RESULTS',tmp_path)
    source=tmp_path/'source.json'; source.write_text('{"v":1}')
    primary=tmp_path/'primary.json';primary.write_text('{}')
    secondary=tmp_path/'secondary.json';secondary.write_text('{}')
    dependency=driver.Stage('input','test',[],source)
    stage=driver.Stage('test','test',[],primary,depends_on=['input'],additional_outputs=[secondary])
    lookup={'input':dependency,'test':stage}
    stamp=driver._stage_stamp(stage);stamp.parent.mkdir()
    stamp.write_text(json.dumps(driver._stage_signature(stage,lookup)))
    assert driver._is_complete(stage,lookup)
    secondary.unlink()
    assert not driver._is_complete(stage,lookup)
    secondary.write_text('{}')
    source.write_text('{"v":2}')
    assert not driver._is_complete(stage,lookup)


def test_dummy_never_falls_back_to_real_json(tmp_path,monkeypatch):
    from utils.paths import resolve_data
    monkeypatch.setenv('THESIS_DUMMY','1')
    real=tmp_path/'real.json';real.write_text('{"scientific":true}')
    assert resolve_data(real) != real
    assert not resolve_data(real).exists()


def test_corrupt_secondary_json_is_not_success(tmp_path):
    primary=tmp_path/'summary.json';primary.write_text('{"status":"OK"}')
    secondary=tmp_path/'measurements.json';secondary.write_text('not json')
    stage=driver.Stage('test','test',[],primary,additional_outputs=[secondary])
    assert 'unparseable JSON' in driver._validate_output(stage)
