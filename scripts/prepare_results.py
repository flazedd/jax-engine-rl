"""Verify and unpack the published experiment data without third-party packages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reproduction' / 'data'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(data: Path, destination: Path, checkpoints: bool = False) -> int:
    """Validate every selected archive before writing; never replace differing files."""
    manifest = json.loads((data / 'manifest.json').read_text())
    selected = ['saved-results.tar.gz']
    if checkpoints:
        selected.append('checkpoints.tar.gz')
    destination = destination.resolve()
    plans = []
    for name in selected:
        spec = manifest['archives'][name]
        archive = data / name
        if digest(archive) != spec['sha256']:
            raise ValueError(f'Archive checksum mismatch: {archive}')
        with tarfile.open(archive, 'r:gz') as source:
            seen = set()
            for member in source:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                        or member.name in seen or member.name not in spec['files_sha256']):
                    raise ValueError(f'Unexpected or unsafe archive member: {member.name}')
                seen.add(member.name)
                target = destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError(f'Destination escapes results directory: {target}')
                with source.extractfile(member) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                expected = spec['files_sha256'][member.name]
                if actual != expected:
                    raise ValueError(f'File checksum mismatch: {member.name}')
                if target.exists() and (not target.is_file() or digest(target) != expected):
                    raise ValueError(f'Refusing to replace different existing data: {target}. '
                                     'Use --destination with a new directory.')
            if seen != set(spec['files_sha256']):
                raise ValueError(f'Incomplete archive: {name}')
        plans.append(archive)
    written = 0
    for archive in plans:
        with tarfile.open(archive, 'r:gz') as source:
            for member in source:
                target = destination / member.name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, target.open('xb') as output:
                    import shutil
                    shutil.copyfileobj(stream, output)
                written += 1
    print(f'Verified {len(selected)} archives; installed {written} files in {destination}')
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'results')
    parser.add_argument('--checkpoints', action='store_true',
                        help='Also unpack the saved training checkpoints for policy replay')
    args = parser.parse_args()
    prepare(DATA, args.destination, args.checkpoints)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
