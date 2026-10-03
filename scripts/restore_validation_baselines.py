"""Restore the explicitly archived, original three-seed M4 validation results.

These are historical validation inputs, not results of the current factorial run.
Their complete per-seed final returns are included in the pinned JSON.
"""
import hashlib
import json
from pathlib import Path
from utils.paths import foundations_dir

SOURCE = Path(__file__).resolve().parents[1] / 'reproduction/inputs/method_ranking.json'
EXPECTED = '277a197bb1da90f519e9401bf36fad9d0a7b6eb9048c35d39b1a6ebaf50480f3'


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Archived validation input checksum mismatch')
    data = json.loads(raw)
    assert len(data['key_stats']['rows']) == 9
    target = foundations_dir() / 'method_ranking.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    print(f'Restored pinned historical validation results: {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
