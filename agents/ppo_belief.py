"""Belief-PPO.

Thin subclass of PPOAgent declaring `requires_analytical_posterior = True`.
The analytical HMM posterior augmentation happens in the env wrapper
`envs/wrappers/belief_obs.py`; this class is the agent-side marker so
registries and leak tests can identify the method.
"""
from __future__ import annotations

from dataclasses import dataclass

from agents.ppo import PPOAgent


@dataclass(frozen=True)
class PPOBeliefAgent(PPOAgent):
    requires_analytical_posterior: bool = True
    produces_belief_for_eval: bool = True
