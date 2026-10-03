"""Run preflight from an archive-free checkout and assert publication isolation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from utils.paths import analysis_dir

ROOT=Path(__file__).resolve().parents[1]


def main():
    clean=Path(tempfile.mkdtemp(prefix='thesis-cleanroom-'))
    for name in ('agents','beliefs','envs','evaluation','experiments','oracles','plotting',
                 'scripts','tests','training','utils','reproduction'):
        shutil.copytree(ROOT/name,clean/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock','README.md','REPRODUCE.md'):
        shutil.copy2(ROOT/name,clean/name)
    protected=clean/'protected_publication';protected.mkdir()
    (protected/'sentinel.txt').write_text('Real publication must remain untouched')
    def snapshot():
        return {str(p.relative_to(protected)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in protected.rglob('*') if p.is_file()}
    before=snapshot()
    env=dict(os.environ,THESIS_RESULTS_ROOT=str(protected/'results'),
             THESIS_PROJECT_FIGS=str(protected/'project_figures'),THESIS_FIG_ROOT=str(protected/'figures'))
    with (clean/'preflight.log').open('w') as log:
        process=subprocess.run([sys.executable,'-m','scripts.run_matched_programme','--dummy'],
                               cwd=clean,env=env,stdout=log,stderr=subprocess.STDOUT)
    unchanged=before==snapshot() and not (clean/'results').exists()
    if process.returncode or not unchanged:
        raise RuntimeError(f'Clean-room preflight failed; inspect {clean}/preflight.log')
    log=(clean/'preflight.log').read_text()
    output=analysis_dir()/'preflight_verification.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'status':'OK','no_archived_results':True,
                                  'inherited_publication_paths_untouched':unchanged,
                                  'log':log},indent=2))
    print(f'Archive-free preflight passed; publication roots untouched. Log: {clean}/preflight.log')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
