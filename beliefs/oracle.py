"""Ground-truth regime extraction from env_state.

This is the single sanctioned module for reading the true latent regime.
Only Oracle-PPO, Belief-PPO (for the analytical HMM forward algorithm it
uses as input, not as ground truth), and per-regime PPO training machinery
should import it. `tests/test_leak.py` verifies that the regime does not
appear in the observation tensor of any other agent.
"""
from __future__ import annotations

import chex


def regime_from_state(state: chex.ArrayTree) -> chex.Array:
    """Return the current latent regime index from an MM env state pytree."""
    return state["regime"]
