"""Oracle-PPO.

Thin subclass of PPOAgent that declares `requires_regime_label = True`.
The actual regime one-hot augmentation happens in `envs/wrappers/oracle_obs.py`;
this class exists so registries and leak tests can identify "this agent is
allowed to see the regime" without inspecting env metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

from agents.ppo import PPOAgent


@dataclass(frozen=True)
class PPOOracleAgent(PPOAgent):
    requires_regime_label: bool = True
