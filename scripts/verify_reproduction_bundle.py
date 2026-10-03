"""Verify all files in an extracted companion archive using only Python's stdlib."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'MANIFEST.json').read_text())
    errors=[]
    for name,expected in manifest['files_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f'missing or unsafe: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            errors.append(f'checksum mismatch: {name}')
    if errors:
        raise ValueError('\n'.join(errors))
    print(f"Verified {len(manifest['files_sha256'])} archived files against SHA-256 manifest.")
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    verify(parser.parse_args().root)
