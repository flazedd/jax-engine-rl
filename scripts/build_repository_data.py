"""Maintainer utility: package published result inputs and checkpoints separately."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from scripts.post_training_evaluation import METHODS
from scripts.prepare_results import digest
from utils.paths import results_root

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> dict:
    root = results_root()
    folders = [root / 'analysis', root / 'foundations']
    folders += [root / 'medium' / experiment for _, experiment in METHODS]
    folders += sorted(root.glob('m5_factorial_*'))
    folders += sorted((root / '_archive').glob('m4_*'))
    files = sorted({p for folder in folders for p in folder.rglob('*')
                    if p.is_file() and not p.is_symlink()
                    and p.suffix in ('.json', '.pkl', '.png') and '.dummy.' not in p.name})
    for _, experiment in METHODS:
        found = {p.name for p in files if p.parent == root / 'medium' / experiment and p.suffix == '.pkl'}
        expected = {f'checkpoint_seed_{seed}.pkl' for seed in range(20)}
        if found != expected:
            raise ValueError(f'Incomplete checkpoint set for {experiment}')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1,
        'description': 'Saved evidence for Trading in the Dark; extract relative to the results directory.',
        'training_commit': '220b319f367e69a8f466c87d5346b1acf60574d6',
        'source_commit_at_packaging': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_identity': 'The Git commit containing this manifest identifies the reproduction code. '
                           'Archive and file hashes identify the saved evidence.',
        'archives': {},
    }
    for name, checkpoint in [('saved-results.tar.gz', False), ('checkpoints.tar.gz', True)]:
        selected = [p for p in files if (p.suffix == '.pkl') == checkpoint]
        path = destination / name
        with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for p in selected:
                    info = archive.gettarinfo(str(p), arcname=p.relative_to(root).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    with p.open('rb') as stream:
                        archive.addfile(info, stream)
        manifest['archives'][name] = {
            'sha256': digest(path), 'bytes': path.stat().st_size,
            'files_sha256': {p.relative_to(root).as_posix(): digest(p) for p in selected},
        }
        print(f'{name}: {len(selected)} files, {path.stat().st_size / 1e6:.1f} MB')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduction' / 'data')
    args = parser.parse_args()
    build(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
