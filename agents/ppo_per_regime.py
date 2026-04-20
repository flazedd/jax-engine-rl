"""Per-regime PPO.

Structurally identical to vanilla PPO; semantics are that it is trained on
an env whose `lock_regime` is fixed. Declaring a separate class makes the
intent visible in configs and lets tests verify no regime-switching was
active at training time. No new loss; reuses training/ppo_update.
"""
from __future__ import annotations

from dataclasses import dataclass

from agents.ppo import PPOAgent


@dataclass(frozen=True)
class PPOPerRegimeAgent(PPOAgent):
    requires_regime_label: bool = True  # receives it via env lock, not obs
