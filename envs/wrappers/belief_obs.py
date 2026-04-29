"""BeliefObsEnv — env-agnostic analytical-posterior wrapper.

Wraps any regime-switching env that exposes:
- `n_regimes: int`
- `initial_distribution: tuple` of length `n_regimes`
- `transition_matrix: tuple` of length `n_regimes ** 2` (row-major)
- `info["regime_likelihood"]: shape [n_regimes]` from each step

The wrapper maintains the analytical HMM posterior in env_state and
appends it to the observation. Update per step:
    b_filt(r) ∝ b(r) · P(evidence_t | regime=r)
    b_pred(r') = sum_r T(r → r') · b_filt(r)
On `done`, belief is reset to the env's `initial_distribution`.

The agent sees `obs = concat(base_obs, belief_over_regimes)`. This is
what Belief-PPO consumes.

D2 ablation: if `constant_belief=True`, the agent sees the env's
`initial_distribution` at every step instead of the live posterior.
Internal belief tracking still runs so logging stays valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax.numpy as jnp


def _initial_belief(env: Any) -> chex.Array:
    return jnp.asarray(env.initial_distribution, dtype=jnp.float32)


def _transition_matrix(env: Any) -> chex.Array:
    n = env.n_regimes
    return jnp.asarray(env.transition_matrix, dtype=jnp.float32).reshape(n, n)


def _filter(belief: chex.Array, likelihood: chex.Array) -> chex.Array:
    post = belief * likelihood
    total = jnp.sum(post)
    return jnp.where(total > 0, post / jnp.maximum(total, 1e-12), belief)


@dataclass(frozen=True)
class BeliefObsEnv:
    inner: Any  # any env exposing the regime-belief support contract above
    constant_belief: bool = False

    @property
    def obs_size(self) -> int:
        return self.inner.obs_size + max(1, self.inner.n_regimes)

    @property
    def n_actions(self) -> int:
        return self.inner.n_actions

    @property
    def episode_length(self) -> int:
        return self.inner.episode_length

    @property
    def gamma(self) -> float:
        return getattr(self.inner, "gamma", 0.99)

    # Pass-through that several MM-specific scripts rely on. Any env that
    # exposes `n_inventory_states` will continue to work; envs without it
    # (e.g. cartpole) won't be hit by those scripts.
    @property
    def n_inventory_states(self) -> int:
        return self.inner.n_inventory_states

    def _augment(self, base_obs: chex.Array, belief: chex.Array) -> chex.Array:
        if self.constant_belief:
            belief = _initial_belief(self.inner)
        return jnp.concatenate([base_obs, belief], axis=-1)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        inner_state, base_obs = self.inner.reset(key)
        b = _initial_belief(self.inner)
        state = {**inner_state, "belief": b}
        return state, self._augment(base_obs, b)

    def step(
        self, state: chex.ArrayTree, action: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        # Strip our auxiliary "belief" key out before delegating to inner
        # so the inner env sees only the keys it expects.
        inner_state = {k: v for k, v in state.items() if k != "belief"}
        new_inner, base_obs, reward, done, info = self.inner.step(
            inner_state, action, key
        )

        # Bayesian filter step: posterior ∝ prior × likelihood.
        b_filt = _filter(state["belief"], info["regime_likelihood"])
        # Prediction step for next regime: b_pred(r') = b_filt @ T.
        T = _transition_matrix(self.inner)
        b_pred = b_filt @ T
        b_reset = _initial_belief(self.inner)
        new_belief = jnp.where(done, b_reset, b_pred)

        new_state = {**new_inner, "belief": new_belief}
        obs = self._augment(base_obs, new_belief)
        info = {**info, "belief": new_belief}
        return new_state, obs, reward, done, info
