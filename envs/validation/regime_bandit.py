"""Regime-switching Bernoulli bandit (validation env for M4).

Stripped-down analogue of the MM env: a latent HMM regime governs the arm
reward probabilities, and the regime evolves *within* an episode according
to a sticky transition matrix. Unlike the canonical RL² bandit (where arm
probs are drawn once per episode and held fixed), this env requires online
filtering of the latent state — the agent must continuously infer which
regime is currently active from the (action, reward) trace.

Default config: 2 regimes × 2 arms, regime 0 favors arm 0 with probs
`[0.9, 0.1]`, regime 1 favors arm 1 with `[0.1, 0.9]`. Sticky transitions
(stay 0.95, switch 0.05). Episode length 100. The marginal-best policy
(no history) earns 0.5/step → 50/episode; perfect regime tracking earns
0.9/step → 90/episode.

This is the M4 test bed for whether RL² / VariBAD can do online filtering
before scaling to MM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chex
import jax
import jax.numpy as jnp


def _default_arm_probs() -> tuple[tuple[float, ...], ...]:
    return ((0.9, 0.1), (0.1, 0.9))


@dataclass(frozen=True)
class RegimeBanditEnv:
    n_regimes: int = 2
    n_arms: int = 2
    episode_length: int = 100
    stay_prob: float = 0.95
    arm_probs: tuple[tuple[float, ...], ...] = field(default_factory=_default_arm_probs)

    @property
    def obs_size(self) -> int:
        return 1

    @property
    def n_actions(self) -> int:
        return self.n_arms

    def _zero_obs(self) -> chex.Array:
        return jnp.zeros((self.obs_size,), dtype=jnp.float32)

    def _arm_probs_array(self) -> chex.Array:
        return jnp.asarray(self.arm_probs, dtype=jnp.float32)

    def _transition_matrix(self) -> chex.Array:
        # Symmetric sticky transitions: diagonal = stay_prob, off-diagonals
        # share the remaining mass uniformly.
        K = self.n_regimes
        off = (1.0 - self.stay_prob) / max(K - 1, 1)
        T = jnp.full((K, K), off, dtype=jnp.float32)
        T = T.at[jnp.arange(K), jnp.arange(K)].set(jnp.float32(self.stay_prob))
        return T

    def _initial_regime(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, (), 0, self.n_regimes).astype(jnp.int32)

    def _next_regime(self, key: chex.PRNGKey, regime: chex.Array) -> chex.Array:
        T = self._transition_matrix()
        row = T[regime]
        return jax.random.categorical(key, jnp.log(row)).astype(jnp.int32)

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        regime = self._initial_regime(key)
        state = {
            "regime": regime,
            "t": jnp.asarray(0, dtype=jnp.int32),
        }
        return state, self._zero_obs()

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        action = action.astype(jnp.int32)
        k_trans, k_reward, k_reset = jax.random.split(key, 3)

        # Transition regime first; reward depends on the new regime (so the
        # action interacts with the regime that just took effect).
        next_regime = self._next_regime(k_trans, state["regime"])
        probs_row = self._arm_probs_array()[next_regime]
        p = probs_row[action]
        u = jax.random.uniform(k_reward, ())
        reward = (u < p).astype(jnp.float32)

        t_next = state["t"] + 1
        done = t_next >= self.episode_length

        regime_after = jnp.where(done, self._initial_regime(k_reset), next_regime)
        t_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)

        new_state = {"regime": regime_after, "t": t_after}
        info = {
            "regime": next_regime,
            "best_arm": jnp.argmax(probs_row).astype(jnp.int32),
            "best_arm_prob": jnp.max(probs_row),
            "regret_step": jnp.max(probs_row) - p,
        }
        return new_state, self._zero_obs(), reward, done, info
