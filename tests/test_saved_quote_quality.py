"""Guard the common-state estimates used by the current thesis chart."""
import json
from pathlib import Path

import pytest

from scripts.summarize_quote_quality import verify_common_states


@pytest.fixture
def payload():
    path = Path(__file__).resolve().parents[1] / 'reproduction/data/decision_quality_comparisons.json'
    return json.loads(path.read_text())


def test_published_common_sample_recomputes(payload):
    report = verify_common_states(payload)
    assert report['runs_per_configuration'] == 20
    assert round(100 * report['mean_coverage'], 1) == 70.1


@pytest.mark.parametrize('field', ['mean', 'ci95'])
def test_changed_saved_estimate_is_rejected(payload, field):
    estimate = payload['common_states']['configurations']['rl2_hypernet']
    if field == 'mean':
        estimate[field] += .01
    else:
        estimate[field][0] += .01
    with pytest.raises(AssertionError):
        verify_common_states(payload)


def test_duplicate_seed_is_rejected(payload):
    payload['common_states']['rows'][1]['seed'] = payload['common_states']['rows'][0]['seed']
    with pytest.raises(ValueError, match='exactly seeds'):
        verify_common_states(payload)


def test_different_coverage_between_agents_is_rejected(payload):
    payload['common_states']['rows'][0]['coverage']['rl2_hypernet'] -= .01
    with pytest.raises(AssertionError):
        verify_common_states(payload)


@pytest.fixture
def reproduced_output(tmp_path):
    from scripts.reproduce_saved_results import CHAPTER_FIGURES, SUPPORTING_TABLES
    output, thesis = tmp_path / 'output', tmp_path / 'thesis'
    for name in CHAPTER_FIGURES:
        path = output / 'figures/results/RQ1' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'nonempty figure fixture')
    for name in SUPPORTING_TABLES:
        for base in (output, thesis):
            path = base / 'tables' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r'Accuracy & $78.1\%$ & $[-1.8,\ +1.1]$')
    return output, thesis


def test_table_check_accepts_tex_spacing(reproduced_output):
    from scripts.reproduce_saved_results import verify_outputs
    output, thesis = reproduced_output
    path = output / 'tables/full_belief_probe.tex'
    path.write_text(path.read_text().replace(r'\ +', '+'))
    assert len(verify_outputs(output, thesis)) == 14


@pytest.mark.parametrize('replacement', ['$78.2\\%$', '$78.1$'])
def test_table_check_rejects_changed_values_or_units(reproduced_output, replacement):
    from scripts.reproduce_saved_results import verify_outputs
    output, thesis = reproduced_output
    path = output / 'tables/full_belief_probe.tex'
    path.write_text(path.read_text().replace(r'$78.1\%$', replacement))
    with pytest.raises(ValueError, match='differs from thesis'):
        verify_outputs(output, thesis)


def test_missing_main_figure_is_rejected(reproduced_output):
    from scripts.reproduce_saved_results import verify_outputs
    output, thesis = reproduced_output
    (output / 'figures/results/RQ1/decision_quality.pdf').unlink()
    with pytest.raises(ValueError, match='Missing or empty'):
        verify_outputs(output, thesis)
