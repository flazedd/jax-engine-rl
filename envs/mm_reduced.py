"""Reduced-form market-making MDP.

State  — bounded discrete inventory q ∈ {-I_max, …, +I_max}.
Action — 3 discrete quote configurations, parameterized by (bid_spread, ask_spread):
         0 = sym       (b=1, a=1) — both tight
         1 = favor_ask (b=3, a=1) — ask tight → ask fills more → inventory ↓
         2 = favor_bid (b=1, a=3) — bid tight → bid fills more → inventory ↑
Fills  — independent Bernoulli per side. Tight side p_tight, wide side p_wide
         (p_tight > p_wide). Captures equal spread (higher reward for wider
         quote when it does fill).
Reward — (bid_fill * bid_spread + ask_fill * ask_spread) − κ · q_next² .
Bounds — a fill that would push inventory past ±I_max is rejected that step.
         (The simplest bounding model; keeps VI tractable.)

E0 (single regime) is this env with one fill-probability table. E1+ overlays
an HMM regime that modulates (p_tight, p_wide); those live in later milestones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp


# Action ids — shared between env, oracle, diagnostics.
ACTION_SYM = 0
ACTION_FAVOR_ASK = 1  # skew inventory ↓
ACTION_FAVOR_BID = 2  # skew inventory ↑
N_ACTIONS = 3

# Spread tables indexed by action → (bid_spread, ask_spread). Widened side
# pays more but fills less.
_BID_SPREADS = jnp.asarray([1.0, 3.0, 1.0], dtype=jnp.float32)
_ASK_SPREADS = jnp.asarray([1.0, 1.0, 3.0], dtype=jnp.float32)
_TIGHT_MASK_BID = jnp.asarray([True, False, True], dtype=jnp.bool_)
_TIGHT_MASK_ASK = jnp.asarray([True, True, False], dtype=jnp.bool_)


@dataclass(frozen=True)
class MMReducedEnv:
    """Single-regime (E0) reduced-form MM MDP with JAX-pure reset/step."""

    inventory_max: int = 5
    episode_length: int = 128
    p_tight: float = 0.6
    p_wide: float = 0.2
    inventory_penalty: float = 0.01
    gamma: float = 0.99
    reset_inventory_range: int = 0  # initial q drawn uniformly in [-r, +r]

    @property
    def n_inventory_states(self) -> int:
        return 2 * self.inventory_max + 1

    @property
    def obs_size(self) -> int:
        # One-hot over inventory states. The agent sees only inventory in E0.
        return self.n_inventory_states

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    # ---- helpers --------------------------------------------------------

    def _inventory_to_obs(self, q: chex.Array) -> chex.Array:
        idx = q + self.inventory_max
        return jax.nn.one_hot(idx, self.n_inventory_states, dtype=jnp.float32)

    def _fill_probs(self, action: chex.Array) -> tuple[chex.Array, chex.Array]:
        bid_tight = _TIGHT_MASK_BID[action]
        ask_tight = _TIGHT_MASK_ASK[action]
        p_bid = jnp.where(bid_tight, self.p_tight, self.p_wide)
        p_ask = jnp.where(ask_tight, self.p_tight, self.p_wide)
        return p_bid, p_ask

    # ---- API ------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        if self.reset_inventory_range > 0:
            q0 = jax.random.randint(
                key,
                (),
                -self.reset_inventory_range,
                self.reset_inventory_range + 1,
            ).astype(jnp.int32)
        else:
            q0 = jnp.asarray(0, dtype=jnp.int32)
        state = {
            "q": q0,
            "t": jnp.asarray(0, dtype=jnp.int32),
        }
        return state, self._inventory_to_obs(q0)

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        q = state["q"]
        t_next = state["t"] + 1

        action = action.astype(jnp.int32)
        p_bid, p_ask = self._fill_probs(action)
        k_bid, k_ask = jax.random.split(key)
        u_bid = jax.random.uniform(k_bid, ())
        u_ask = jax.random.uniform(k_ask, ())

        bid_fill_raw = u_bid < p_bid
        ask_fill_raw = u_ask < p_ask

        # Reject fills that would push inventory past bounds.
        bid_ok = q < self.inventory_max
        ask_ok = q > -self.inventory_max
        bid_fill = bid_fill_raw & bid_ok
        ask_fill = ask_fill_raw & ask_ok

        bid_spread = _BID_SPREADS[action]
        ask_spread = _ASK_SPREADS[action]
        capture = (
            bid_fill.astype(jnp.float32) * bid_spread
            + ask_fill.astype(jnp.float32) * ask_spread
        )

        q_next = q + bid_fill.astype(jnp.int32) - ask_fill.astype(jnp.int32)
        inv_pen = self.inventory_penalty * jnp.square(q_next.astype(jnp.float32))
        reward = (capture - inv_pen).astype(jnp.float32)

        done = t_next >= self.episode_length
        # On done, reset inventory to 0 and t to 0 so episodes tile cleanly.
        q_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), q_next)
        t_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)

        new_state = {"q": q_after, "t": t_after}
        obs = self._inventory_to_obs(q_after)
        info = {"inventory": q_next}
        return new_state, obs, reward, done, info
