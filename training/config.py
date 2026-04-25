"""Experiment configuration.

One YAML file per experiment, composable via `extends:`. Loaded into a typed
dataclass at startup; run-mode flags mutate the dataclass in memory. After
resolution the config is serialized to JSON for reproducibility.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"


@dataclass
class EnvConfig:
    name: str = "dummy"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    name: str = "dummy"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str
    env: EnvConfig
    agent: AgentConfig
    iterations: int = 100
    parallel_envs: int = 512
    rollout_length: int = 128
    num_seeds: int = 5
    seed_base: int = 0
    run_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML loading with `extends:` composition
# ---------------------------------------------------------------------------


def _resolve_path(raw: str, relative_to: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate = (relative_to / p).resolve()
    if candidate.exists():
        return candidate
    return (CONFIG_ROOT / p).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"circular extends chain involving {path}")
    seen = seen | {path}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ValueError(f"{path}: extends must be a string or list of strings")

    merged: dict[str, Any] = {}
    for ext in extends:
        ext_path = _resolve_path(ext, path.parent)
        merged = _deep_merge(merged, _load_yaml_with_extends(ext_path, seen))

    return _deep_merge(merged, raw)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        path = CONFIG_ROOT / path
    raw = _load_yaml_with_extends(path)

    env_raw = raw.get("env", {}) or {}
    agent_raw = raw.get("agent", {}) or {}

    if "experiment_name" not in raw:
        raise ValueError(f"{path}: missing required field 'experiment_name'")

    cfg = ExperimentConfig(
        experiment_name=raw["experiment_name"],
        env=EnvConfig(
            name=env_raw.get("name", "dummy"),
            params=env_raw.get("params", {}) or {},
        ),
        agent=AgentConfig(
            name=agent_raw.get("name", "dummy"),
            params=agent_raw.get("params", {}) or {},
        ),
        iterations=int(raw.get("iterations", 100)),
        parallel_envs=int(raw.get("parallel_envs", 512)),
        rollout_length=int(raw.get("rollout_length", 128)),
        num_seeds=int(raw.get("num_seeds", 5)),
        seed_base=int(raw.get("seed_base", 0)),
    )
    return cfg


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


def apply_run_mode(cfg: ExperimentConfig, mode: str) -> ExperimentConfig:
    """Mutate `cfg` in place with run-mode overrides and return it.

    - super_fast: 2 iters, 16 envs, rollout 32, 1 seed — pipeline smoke test.
    - fast:       20 iters, 128 envs, rollout 128, 1 seed — directional signal.
    - mid:        100 iters, 256 envs, rollout 128, 1 seed — single-seed plateau,
                  enough to read the converged gap when iterating env designs.
    - full:       config defaults — production run.
    """
    if mode == "super_fast":
        cfg.iterations = 2
        cfg.parallel_envs = 16
        cfg.rollout_length = 32
        cfg.num_seeds = 1
    elif mode == "fast":
        cfg.iterations = 20
        cfg.parallel_envs = 128
        cfg.rollout_length = 128
        cfg.num_seeds = 1
    elif mode == "mid":
        cfg.iterations = 100
        cfg.parallel_envs = 256
        cfg.rollout_length = 128
        cfg.num_seeds = 1
    elif mode == "full":
        pass
    else:
        raise ValueError(f"unknown run mode: {mode!r}")
    cfg.run_mode = mode
    return cfg


def dump_config_json(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)
