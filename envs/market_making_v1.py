"""Reduced-form market-making MDP, with optional regime-switching (E1+).

State  — bounded discrete inventory q ∈ {-I_max, …, +I_max}, latent regime r.
Action — 3 discrete quote configurations, parameterized by (bid_spread, ask_spread):
         0 = sym       (b=1, a=1) — both tight
         1 = favor_ask (b=3, a=1) — ask tight → ask fills more → inventory ↓
         2 = favor_bid (b=1, a=3) — bid tight → bid fills more → inventory ↑
Fills  — independent Bernoulli per side, with per-regime (p_tight, p_wide)
         on bid and ask sides separately. For E0 (single regime) the scalar
         p_tight / p_wide apply to both sides.
Reward — (bid_fill * bid_spread + ask_fill * ask_spread) − κ · q_next² .
Bounds — a fill that would push inventory past ±I_max is rejected that step.
Regime — latent HMM with `n_regimes` states and a row-stochastic transition
         matrix. Initial regime is sampled from `initial_distribution`.
         `lock_regime >= 0` freezes the regime for the whole episode (used
         for per-regime PPO and R1/R2 diagnostics).

Observation: one-hot over inventory states only. The regime is *not* in obs;
it is accessible only through `beliefs/oracle.py` extraction from env_state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
class MarketMakingV1:
    """Reduced-form MM MDP with JAX-pure reset/step.

    Single-regime (E0) mode: leave all regime fields at defaults and rely on
    the scalar `p_tight` / `p_wide`. Regime-switching (E1+) mode: set
    `n_regimes > 1` and populate the per-regime fill probability tuples and
    transition matrix.

    Every list-valued field is coerced to a tuple in __post_init__ so YAML
    loading (which gives lists) produces a hashable instance compatible with
    `static_argnames=("env", ...)` in the JIT'd rollout.
    """

    inventory_max: int = 5
    episode_length: int = 128
    p_tight: float = 0.6
    p_wide: float = 0.2
    inventory_penalty: float = 0.01
    gamma: float = 0.99
    reset_inventory_range: int = 0

    # Regime-switching (E1+). n_regimes=1 disables it and keeps the scalar
    # p_tight / p_wide on both bid and ask sides.
    n_regimes: int = 1
    # Row-major 1D tuple of length n_regimes ** 2.
    transition_matrix: tuple = field(default_factory=tuple)
    # Per-regime bid/ask fill probabilities, each length n_regimes.
    regime_p_tight_bid: tuple = field(default_factory=tuple)
    regime_p_wide_bid: tuple = field(default_factory=tuple)
    regime_p_tight_ask: tuple = field(default_factory=tuple)
    regime_p_wide_ask: tuple = field(default_factory=tuple)
    initial_distribution: tuple = field(default_factory=tuple)
    # -1 → regime transitions freely; >=0 → regime locked at that index.
    lock_regime: int = -1

    def __post_init__(self) -> None:
        # Coerce YAML lists to tuples so frozen + hashable.
        for name in (
            "transition_matrix",
            "regime_p_tight_bid",
            "regime_p_wide_bid",
            "regime_p_tight_ask",
            "regime_p_wide_ask",
            "initial_distribution",
        ):
            v = getattr(self, name)
            if not isinstance(v, tuple):
                object.__setattr__(self, name, tuple(v))

        if self.n_regimes > 1:
            n = self.n_regimes
            if len(self.transition_matrix) != n * n:
                raise ValueError(
                    f"transition_matrix must have length {n*n}, got {len(self.transition_matrix)}"
                )
            for name in (
                "regime_p_tight_bid",
                "regime_p_wide_bid",
                "regime_p_tight_ask",
                "regime_p_wide_ask",
                "initial_distribution",
            ):
                v = getattr(self, name)
                if len(v) != n:
                    raise ValueError(f"{name} must have length {n}, got {len(v)}")

    # ---- properties -----------------------------------------------------

    @property
    def n_inventory_states(self) -> int:
        return 2 * self.inventory_max + 1

    @property
    def obs_size(self) -> int:
        # One-hot over inventory states. The regime is intentionally hidden.
        return self.n_inventory_states

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    # ---- helpers --------------------------------------------------------

    def _inventory_to_obs(self, q: chex.Array) -> chex.Array:
        idx = q + self.inventory_max
        return jax.nn.one_hot(idx, self.n_inventory_states, dtype=jnp.float32)

    def _fill_probs(
        self, action: chex.Array, regime: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        bid_tight = _TIGHT_MASK_BID[action]
        ask_tight = _TIGHT_MASK_ASK[action]
        if self.n_regimes == 1:
            p_bid = jnp.where(bid_tight, self.p_tight, self.p_wide)
            p_ask = jnp.where(ask_tight, self.p_tight, self.p_wide)
            return p_bid.astype(jnp.float32), p_ask.astype(jnp.float32)

        pt_bid = jnp.asarray(self.regime_p_tight_bid, dtype=jnp.float32)[regime]
        pw_bid = jnp.asarray(self.regime_p_wide_bid, dtype=jnp.float32)[regime]
        pt_ask = jnp.asarray(self.regime_p_tight_ask, dtype=jnp.float32)[regime]
        pw_ask = jnp.asarray(self.regime_p_wide_ask, dtype=jnp.float32)[regime]
        p_bid = jnp.where(bid_tight, pt_bid, pw_bid)
        p_ask = jnp.where(ask_tight, pt_ask, pw_ask)
        return p_bid, p_ask

    def _sample_regime(self, key: chex.PRNGKey) -> chex.Array:
        if self.lock_regime >= 0:
            return jnp.asarray(self.lock_regime, dtype=jnp.int32)
        if self.n_regimes == 1:
            return jnp.asarray(0, dtype=jnp.int32)
        probs = jnp.asarray(self.initial_distribution, dtype=jnp.float32)
        return jax.random.choice(key, self.n_regimes, p=probs).astype(jnp.int32)

    def _transition_regime(
        self, regime: chex.Array, key: chex.PRNGKey
    ) -> chex.Array:
        if self.n_regimes == 1 or self.lock_regime >= 0:
            return regime
        mat = jnp.asarray(self.transition_matrix, dtype=jnp.float32).reshape(
            self.n_regimes, self.n_regimes
        )
        row = mat[regime]
        return jax.random.choice(key, self.n_regimes, p=row).astype(jnp.int32)

    # ---- API ------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        k_inv, k_reg = jax.random.split(key)
        if self.reset_inventory_range > 0:
            q0 = jax.random.randint(
                k_inv,
                (),
                -self.reset_inventory_range,
                self.reset_inventory_range + 1,
            ).astype(jnp.int32)
        else:
            q0 = jnp.asarray(0, dtype=jnp.int32)
        r0 = self._sample_regime(k_reg)
        state = {
            "q": q0,
            "t": jnp.asarray(0, dtype=jnp.int32),
            "regime": r0,
        }
        return state, self._inventory_to_obs(q0)

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        q = state["q"]
        regime = state["regime"]
        t_next = state["t"] + 1

        action = action.astype(jnp.int32)
        p_bid, p_ask = self._fill_probs(action, regime)
        k_bid, k_ask, k_reg, k_init = jax.random.split(key, 4)
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

        # Regime transition happens every step (continuously switching HMM).
        regime_next = self._transition_regime(regime, k_reg)

        done = t_next >= self.episode_length
        # On done, reset inventory to 0, t to 0, regime resampled.
        q_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), q_next)
        t_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)
        regime_after = jnp.where(done, self._sample_regime(k_init), regime_next)

        new_state = {"q": q_after, "t": t_after, "regime": regime_after}
        obs = self._inventory_to_obs(q_after)
        # info carries eval-only fields (regime, fills). The training loop
        # (rollout.py) strips info before the agent sees the trajectory — see
        # beliefs/oracle.py for the sanctioned read path.
        info = {
            "inventory": q_next,
            "regime": regime,          # regime during the step just taken
            "regime_next": regime_next,
            "bid_fill": bid_fill.astype(jnp.int32),
            "ask_fill": ask_fill.astype(jnp.int32),
        }
        return new_state, obs, reward, done, info
