"""BeliefObsEnv — wraps an MarketMakingV1, appends the analytical HMM posterior
to the observation.

The posterior is tracked in env_state and updated at every step using
`beliefs.hmm_posterior.full_update`. On episode boundaries (done=True) the
belief resets to the env's `initial_distribution`.

The agent sees `obs = concat(inventory_one_hot, belief_over_regimes)`. This
is what Belief-PPO consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp

from beliefs.hmm_posterior import full_update, initial_belief
from envs.market_making_v1 import MarketMakingV1


@dataclass(frozen=True)
class BeliefObsEnv:
    inner: MarketMakingV1
    # D2 ablation: if True, agent sees `initial_distribution` at every step.
    # Internal belief tracking still runs so logging is unaffected.
    constant_belief: bool = False

    @property
    def n_inventory_states(self) -> int:
        return self.inner.n_inventory_states

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
        return self.inner.gamma

    def _augment(self, base_obs: chex.Array, belief: chex.Array) -> chex.Array:
        if self.constant_belief:
            belief = initial_belief(self.inner)
        return jnp.concatenate([base_obs, belief], axis=-1)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        inner_state, base_obs = self.inner.reset(key)
        b = initial_belief(self.inner)
        state = {**inner_state, "belief": b}
        return state, self._augment(base_obs, b)

    def step(
        self, state: chex.ArrayTree, action: chex.Array, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        # The inner state keys expected by inner.step are {"q", "t", "regime"}.
        inner_keys = {"q": state["q"], "t": state["t"], "regime": state["regime"]}
        new_inner, base_obs, reward, done, info = self.inner.step(
            inner_keys, action, key
        )

        # Filter the belief using the (action, fills, q) observed this step.
        # q is the inventory at the *start* of the step (before fills resolved),
        # which is exactly state["q"]. That is what the likelihood uses to mask
        # blocked sides.
        _, b_pred = full_update(
            state["belief"],
            self.inner,
            action,
            info["bid_fill"],
            info["ask_fill"],
            state["q"],
        )
        # On episode boundary, reset belief to prior.
        b_reset = initial_belief(self.inner)
        new_belief = jnp.where(done, b_reset, b_pred)

        new_state = {**new_inner, "belief": new_belief}
        obs = self._augment(base_obs, new_belief)
        # Expose belief in info for downstream logging (entropy plots).
        info = {**info, "belief": new_belief}
        return new_state, obs, reward, done, info
