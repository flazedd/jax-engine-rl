"""Collect the inexpensive legacy invariant checks in the normal pytest run."""
import importlib
from pathlib import Path
import pytest

MODULES = ('test_beliefs', 'test_envs', 'test_stack_obs', 'test_bandit', 'test_gridworld', 'test_regime_bandit')
CASES = [(name, fn_name) for name in MODULES
         for fn_name in vars(importlib.import_module('tests.' + name)) if fn_name.startswith('_test_')]


@pytest.mark.parametrize('module_name,function_name', CASES)
def test_legacy_invariant(module_name, function_name):
    module=importlib.import_module('tests.'+module_name)
    args=(module._load_env(),) if module_name == 'test_beliefs' else ()
    assert getattr(module, function_name)(*args) == 0


@pytest.mark.parametrize('module_name', ('test_envs','test_beliefs'))
def test_final_e9_invariants(module_name, monkeypatch):
    module=importlib.import_module('tests.'+module_name)
    monkeypatch.setattr(module,'E_FINAL',Path(__file__).resolve().parents[1]/'experiments/configs/envs/e9_rare_fills.yaml')
    for name in vars(module):
        if name.startswith('_test_'):
            args=(module._load_env(),) if module_name == 'test_beliefs' else ()
            assert getattr(module,name)(*args) == 0
