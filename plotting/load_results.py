"""JSON parsing helpers for result directories.

All plotting code reads JSON via these helpers (never from training in-memory
state), enforcing the "JSON is the contract" rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_metrics(experiment_dir: str | Path) -> dict[str, Any]:
    d = Path(experiment_dir)
    return _read_json(d / "metrics.json")


def load_summary(experiment_dir: str | Path) -> dict[str, Any]:
    d = Path(experiment_dir)
    return _read_json(d / "summary.json")


def load_eval(experiment_dir: str | Path) -> dict[str, Any]:
    d = Path(experiment_dir)
    return _read_json(d / "eval.json")


def load_config(experiment_dir: str | Path) -> dict[str, Any]:
    d = Path(experiment_dir)
    return _read_json(d / "config.json")
