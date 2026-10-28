"""Published data must verify before extraction and must preserve local results."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.prepare_results import prepare, digest


def archive(tmp_path, name='analysis/result.json', payload=b'{"result": 1}', expected=None):
    data = tmp_path / 'data'
    data.mkdir()
    path = data / 'saved-results.tar.gz'
    with tarfile.open(path, 'w:gz') as output:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))
    spec = {'sha256': digest(path), 'files_sha256': {
        name: expected or hashlib.sha256(payload).hexdigest()}}
    (data / 'manifest.json').write_text(json.dumps({'archives': {'saved-results.tar.gz': spec}}))
    return data


def test_install_and_repeat(tmp_path):
    data = archive(tmp_path)
    target = tmp_path / 'results'
    assert prepare(data, target) == 1
    assert prepare(data, target) == 0
    assert json.loads((target / 'analysis/result.json').read_text()) == {'result': 1}


def test_changed_local_result_is_preserved(tmp_path):
    data = archive(tmp_path)
    target = tmp_path / 'results'
    prepare(data, target)
    result = target / 'analysis/result.json'
    result.write_text('local experiment')
    with pytest.raises(ValueError, match='Refusing to replace'):
        prepare(data, target)
    assert result.read_text() == 'local experiment'


@pytest.mark.parametrize('member', ['../outside.json', '/absolute.json'])
def test_reject_unsafe_paths_before_writing(tmp_path, member):
    data = archive(tmp_path, name=member)
    with pytest.raises(ValueError, match='unsafe'):
        prepare(data, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


def test_detect_corrupt_file_before_writing(tmp_path):
    data = archive(tmp_path, expected='0' * 64)
    with pytest.raises(ValueError, match='File checksum mismatch'):
        prepare(data, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


def test_detect_corrupt_archive_before_writing(tmp_path):
    data = archive(tmp_path)
    with (data / 'saved-results.tar.gz').open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError, match='Archive checksum mismatch'):
        prepare(data, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


def test_thesis_table_can_move_to_appendix_without_losing_checks(tmp_path):
    from scripts.thesis_contract import _thesis_table_text
    sections = tmp_path / 'sections'
    sections.mkdir()
    table = r'\begin{table}[H] Method & $+2.31$ \\ \label{tab:integration_gap}\end{table}'
    (sections / 'results.tex').write_text(table)
    expected = _thesis_table_text(tmp_path, 'tab:integration_gap')
    (sections / 'results.tex').write_text('See the appendix.')
    (sections / 'appendix.tex').write_text(table)
    assert _thesis_table_text(tmp_path, 'tab:integration_gap') == expected
    (sections / 'appendix.tex').write_text(table.replace('+2.31', '+9.99'))
    assert _thesis_table_text(tmp_path, 'tab:integration_gap') != expected
    (sections / 'results.tex').write_text(table)
    assert _thesis_table_text(tmp_path, 'tab:integration_gap') == ''


@pytest.mark.parametrize('problem', ['corrupt', 'missing_manifest_entry', 'missing_file'])
def test_supplemental_inputs_verify_before_extraction(tmp_path, problem):
    from scripts.prepare_results import SUPPLEMENTAL_FILES
    data = archive(tmp_path)
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = 2
    manifest['supplemental_files_sha256'] = {}
    for name in SUPPLEMENTAL_FILES:
        (data / name).write_text('{}')
        manifest['supplemental_files_sha256'][name] = digest(data / name)
    if problem == 'corrupt':
        (data / SUPPLEMENTAL_FILES[0]).write_text('{"changed": true}')
    elif problem == 'missing_manifest_entry':
        del manifest['supplemental_files_sha256'][SUPPLEMENTAL_FILES[0]]
    else:
        (data / SUPPLEMENTAL_FILES[0]).unlink()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(data, tmp_path / 'results')
    assert not (tmp_path / 'results').exists()


def test_verified_supplemental_inputs_allow_extraction(tmp_path):
    from scripts.prepare_results import SUPPLEMENTAL_FILES
    data = archive(tmp_path)
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = 2
    manifest['supplemental_files_sha256'] = {}
    for name in SUPPLEMENTAL_FILES:
        (data / name).write_text('{}')
        manifest['supplemental_files_sha256'][name] = digest(data / name)
    manifest_path.write_text(json.dumps(manifest))
    assert prepare(data, tmp_path / 'results') == 1
