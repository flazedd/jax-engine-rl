"""Fail-closed identity and integrity checks for newly trained artifacts.

Legacy checkpoints are evidence for the archived thesis, not resumable caches.
They must be replayed separately or retrained under a fresh results root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(config: dict) -> dict:
    config = json.loads(json.dumps(
        {k: v for k, v in config.items() if k != "commit_hash"}, sort_keys=True, default=str))
    files = sorted(p for folder in ("agents", "beliefs", "envs", "oracles", "training", "utils")
                   for p in (ROOT / folder).rglob("*.py"))
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    code = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    payload = {"schema_version": 2, "config": config, "source_sha256": code}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return payload


def validate_seed(folder: Path, seed: int, expected: dict) -> dict:
    cached = json.loads((folder / f"seed_{seed}_result.json").read_text())
    checkpoint = folder / f"checkpoint_seed_{seed}.pkl"
    if cached.get("_training_fingerprint") != expected["fingerprint"]:
        raise ValueError(f"seed {seed}: incompatible or legacy training identity")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"seed {seed}: missing or empty checkpoint")
    if cached.get("_checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"seed {seed}: checkpoint checksum mismatch")
    if cached.get("seed") != seed:
        raise ValueError(f"seed {seed}: incorrect seed identifier")
    iterations = expected["config"]["iterations"]
    for key in ("mean_return_per_iter", "var_return_per_iter", "per_iter_times"):
        values = cached.get(key, [])
        if len(values) != iterations:
            raise ValueError(f"seed {seed}: incomplete {key}")
        import math
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"seed {seed}: non-finite {key}")
    return cached


def prepare(folder: Path, expected: dict) -> None:
    """Validate every existing seed BEFORE any config/provenance writes."""
    artifacts = list(folder.glob("checkpoint_seed_*.pkl")) + list(folder.glob("seed_*_result.json"))
    if artifacts:
        try:
            previous = json.loads((folder / "provenance.json").read_text())
            if previous != expected:
                raise ValueError("training configuration or source differs")
            allowed = set(range(expected["config"]["seed_base"],
                                expected["config"]["seed_base"] + expected["config"]["num_seeds"]))
            found = {int(p.stem.split("_")[2]) if p.name.startswith("checkpoint")
                     else int(p.stem.split("_")[1]) for p in artifacts}
            if not found <= allowed:
                raise ValueError("unexpected seed identifiers")
            for seed in sorted(found):
                validate_seed(folder, seed, expected)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Refusing to alter {folder}: {exc}. "
                             "Use a fresh THESIS_RESULTS_ROOT; preserve the archived run.") from exc
    (folder / "provenance.json").write_text(json.dumps(expected, indent=2, default=str))


def complete(folder: Path, config: dict | None = None) -> bool:
    try:
        saved = json.loads((folder / "config.json").read_text())
        expected = identity(config if config is not None else saved)
        provenance = json.loads((folder / "provenance.json").read_text())
        if provenance != expected:
            return False
        if json.loads((folder / "summary.json").read_text()).get("status") != "OK":
            return False
        cfg = expected["config"]
        ids = set(range(cfg["seed_base"], cfg["seed_base"] + cfg["num_seeds"]))
        if {int(p.stem.split("_")[2]) for p in folder.glob("checkpoint_seed_*.pkl")} != ids:
            return False
        caches = [validate_seed(folder, seed, expected) for seed in sorted(ids)]
        metrics = json.loads((folder / "metrics.json").read_text())
        import numpy as np
        if metrics.get("seeds") != sorted(ids):
            return False
        if not np.array_equal(metrics.get("per_seed_mean_return_per_iter"),
                              [c["mean_return_per_iter"] for c in caches]):
            return False
        return all(metrics.get(k) == cfg[k] for k in
                   ("num_seeds", "iterations", "parallel_envs", "rollout_length"))
    except (OSError, ValueError, KeyError, TypeError):
        return False
