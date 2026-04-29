"""Regime-switching CartPole — second POMDP env for the decoupling thesis.

A continuous-state cartpole with HMM-switching force magnitude. The
agent balances a pole on a cart; on each step it pushes the cart left
(action 0) or right (action 1). The latent regime governs the
*magnitude* of that push, so a successful policy must condition its
control gain on the regime: weak-force regimes need anticipatory or
more-frequent corrections; strong-force regimes need restraint to
avoid overshoot.

Force-magnitude regimes (rather than gravity regimes) keep the
regime signal present at every step regardless of pole angle: the
applied force directly enters θ_ddot via cos(θ), so each observed
transition is informative. A pure-gravity regime variable would
silence the signal whenever the policy successfully holds θ near
zero — exactly the regime R1 (policy divergence) we depend on.

The analytical posterior is a closed-form Bayesian filter on the
angular-velocity-change residual (Gaussian under each regime
hypothesis).

This env satisfies the env-agnostic wrapper contract:
- exposes `n_regimes`, `initial_distribution`, `transition_matrix`
- stores the true regime under `state["regime"]`
- emits `info["regime_likelihood"]: shape [n_regimes]` per step

so it plugs into BeliefObsEnv / OracleObsEnv / RL2ObsEnv / StackObsEnv
without any wrapper changes.

Used as a second-POMDP external-validity probe of the M5 / M6
hypernet ≫ concat decoupling finding. Different physics, different
observation modality, different action space — same regime-switching
HMM structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chex
import jax
import jax.numpy as jnp


# Physical constants (SI units). Cartpole is a standard control
# benchmark; these match the canonical Gym formulation. Gravity is
# regime-conditional and lives in `regime_gravity`.
_DT: float = 0.02              # 50 Hz integration
_POLE_LENGTH_HALF: float = 0.5  # half-length of pole, m
_POLE_MASS: float = 0.1        # kg
_CART_MASS: float = 1.0        # kg
_TOTAL_MASS: float = _POLE_MASS + _CART_MASS
_POLE_MASS_LENGTH: float = _POLE_MASS * _POLE_LENGTH_HALF
_GRAVITY: float = 9.8          # m/s² (Earth; held fixed across regimes)
_MAX_X: float = 2.4            # cart-position failure threshold (m)
_MAX_THETA: float = 12.0 * jnp.pi / 180.0  # pole-angle failure threshold (≈0.209 rad)

# Initialization spread. Pole starts near-upright with small uniform
# perturbations on every state component, matching Gym CartPole.
_INIT_SPREAD: float = 0.05

# Observation normalization scales. Chosen so typical episode states
# put each obs channel roughly in [-1, 1].
_OBS_X_SCALE: float = _MAX_X
_OBS_XDOT_SCALE: float = 5.0
_OBS_THETA_SCALE: float = _MAX_THETA
_OBS_THETADOT_SCALE: float = 5.0

N_ACTIONS: int = 2


@dataclass(frozen=True)
class CartPoleRegimeV1:
    """Cartpole with HMM-switching gravity.

    Frozen dataclass so it is hashable and JIT-compatible (mirrors
    `MarketMakingV1`). All list-valued fields are coerced to tuples in
    `__post_init__` so YAML loading produces hashable instances.

    Process noise (`angular_velocity_noise_std`) is essential: with
    deterministic dynamics the regime would be revealed exactly in one
    step. The noise turns the env into a proper switching state-space
    model where the posterior is a non-degenerate Bayesian filter.
    """

    episode_length: int = 128
    gamma: float = 0.99
    angular_velocity_noise_std: float = 0.2  # σ on θ-dot transition

    # Regime-switching HMM. n_regimes=3 is the project standard.
    n_regimes: int = 3
    # Per-regime push-force magnitude (N). Length n_regimes.
    regime_force_magnitude: tuple = (5.0, 10.0, 20.0)
    # Row-major flat 1D tuple of length n_regimes ** 2.
    transition_matrix: tuple = field(default_factory=tuple)
    initial_distribution: tuple = field(default_factory=tuple)
    # -1 → regime transitions freely; >=0 → regime locked at that index.
    lock_regime: int = -1

    def __post_init__(self) -> None:
        for name in ("regime_force_magnitude", "transition_matrix", "initial_distribution"):
            v = getattr(self, name)
            if not isinstance(v, tuple):
                object.__setattr__(self, name, tuple(v))
        n = self.n_regimes
        if len(self.regime_force_magnitude) != n:
            raise ValueError(
                f"regime_force_magnitude must have length {n}, got "
                f"{len(self.regime_force_magnitude)}"
            )
        if n > 1:
            if len(self.transition_matrix) != n * n:
                raise ValueError(
                    f"transition_matrix must have length {n*n}, got "
                    f"{len(self.transition_matrix)}"
                )
            if len(self.initial_distribution) != n:
                raise ValueError(
                    f"initial_distribution must have length {n}, got "
                    f"{len(self.initial_distribution)}"
                )

    # ---- properties ----------------------------------------------------

    @property
    def obs_size(self) -> int:
        return 4  # (x, x_dot, theta, theta_dot), normalized

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    # ---- helpers -------------------------------------------------------

    def _normalize_obs(self, x, x_dot, theta, theta_dot) -> chex.Array:
        return jnp.stack([
            x / _OBS_X_SCALE,
            jnp.tanh(x_dot / _OBS_XDOT_SCALE),
            theta / _OBS_THETA_SCALE,
            jnp.tanh(theta_dot / _OBS_THETADOT_SCALE),
        ]).astype(jnp.float32)

    def _angular_acceleration(
        self,
        theta: chex.Array,
        theta_dot: chex.Array,
        force: chex.Array,
    ) -> chex.Array:
        """Pole angular acceleration under given applied force. Standard
        cartpole equations with fixed Earth gravity."""
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        temp = (force + _POLE_MASS_LENGTH * theta_dot**2 * sin_t) / _TOTAL_MASS
        thetaacc_num = _GRAVITY * sin_t - cos_t * temp
        thetaacc_den = _POLE_LENGTH_HALF * (
            4.0 / 3.0 - _POLE_MASS * cos_t**2 / _TOTAL_MASS
        )
        return thetaacc_num / thetaacc_den

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

    def _initial_state_components(
        self, key: chex.PRNGKey
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Sample (x, x_dot, theta, theta_dot) near upright (Gym convention)."""
        u = jax.random.uniform(
            key, shape=(4,), minval=-_INIT_SPREAD, maxval=_INIT_SPREAD,
        )
        return u[0], u[1], u[2], u[3]

    def _regime_likelihood(
        self,
        theta: chex.Array,
        theta_dot: chex.Array,
        action: chex.Array,
        observed_theta_dot_next: chex.Array,
    ) -> chex.Array:
        """P(observed θ-dot transition | regime, prev state, action) — [n_regimes].

        Under regime r the agent's push has magnitude F_r so the
        predicted θ-dot one step ahead is θ_dot + θ_ddot(F_r) · dt.
        The observed transition has Gaussian process noise σ², so the
        likelihood is a Gaussian density on the residual. Returned as
        un-normalized probabilities (the Bayesian filter normalizes
        them, so the constant prefactor and global scale cancel). For
        numerical stability we subtract the max log-likelihood before
        exp.
        """
        sign = jnp.where(action == 1, 1.0, -1.0)
        f_per = jnp.asarray(self.regime_force_magnitude, dtype=jnp.float32)
        # Predicted angular acceleration under each regime's force magnitude.
        theta_ddot_per = jax.vmap(
            lambda f: self._angular_acceleration(theta, theta_dot, sign * f)
        )(f_per)
        predicted_dot_next = theta_dot + theta_ddot_per * _DT
        residual = observed_theta_dot_next - predicted_dot_next
        sigma = self.angular_velocity_noise_std
        log_lik = -0.5 * (residual / sigma) ** 2
        log_lik_shifted = log_lik - jnp.max(log_lik)
        return jnp.exp(log_lik_shifted).astype(jnp.float32)

    # ---- API -----------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> tuple[chex.ArrayTree, chex.Array]:
        k_state, k_reg = jax.random.split(key)
        x, x_dot, theta, theta_dot = self._initial_state_components(k_state)
        regime = self._sample_regime(k_reg)
        state = {
            "x": x.astype(jnp.float32),
            "x_dot": x_dot.astype(jnp.float32),
            "theta": theta.astype(jnp.float32),
            "theta_dot": theta_dot.astype(jnp.float32),
            "t": jnp.asarray(0, dtype=jnp.int32),
            "regime": regime,
        }
        return state, self._normalize_obs(state["x"], state["x_dot"],
                                          state["theta"], state["theta_dot"])

    def step(
        self,
        state: chex.ArrayTree,
        action: chex.Array,
        key: chex.PRNGKey,
    ) -> tuple[chex.ArrayTree, chex.Array, chex.Array, chex.Array, dict[str, Any]]:
        action = action.astype(jnp.int32)
        regime = state["regime"]
        force_mag_now = jnp.asarray(
            self.regime_force_magnitude, dtype=jnp.float32
        )[regime]
        force = jnp.where(action == 1, force_mag_now, -force_mag_now)

        # Forward Euler integration under the true (regime-conditional) force.
        theta = state["theta"]
        theta_dot = state["theta_dot"]
        theta_ddot = self._angular_acceleration(theta, theta_dot, force)
        x_ddot = (
            (force + _POLE_MASS_LENGTH * theta_dot**2 * jnp.sin(theta))
            / _TOTAL_MASS
            - _POLE_MASS_LENGTH * theta_ddot * jnp.cos(theta) / _TOTAL_MASS
        )

        # Process noise on angular velocity. This is what makes the posterior
        # non-degenerate — without it the first observed transition reveals
        # the regime exactly.
        k_noise, k_reg, k_init = jax.random.split(key, 3)
        eps = jax.random.normal(k_noise, ()) * self.angular_velocity_noise_std

        x_next = state["x"] + state["x_dot"] * _DT
        x_dot_next = state["x_dot"] + x_ddot * _DT
        theta_next = theta + theta_dot * _DT
        theta_dot_next = theta_dot + theta_ddot * _DT + eps

        # Reward: +1 per step pole upright AND cart on track. No early
        # termination — episode runs the full episode_length.
        upright = jnp.abs(theta_next) < _MAX_THETA
        on_track = jnp.abs(x_next) < _MAX_X
        reward = (upright & on_track).astype(jnp.float32)

        # HMM regime transition.
        regime_next = self._transition_regime(regime, k_reg)

        t_next = state["t"] + 1
        done = t_next >= self.episode_length
        # On done, reset state to a fresh initial draw and resample the regime.
        x_after_components = self._initial_state_components(k_init)
        regime_after = jnp.where(done, self._sample_regime(k_init), regime_next)

        x_after = jnp.where(done, x_after_components[0], x_next).astype(jnp.float32)
        x_dot_after = jnp.where(done, x_after_components[1], x_dot_next).astype(jnp.float32)
        theta_after = jnp.where(done, x_after_components[2], theta_next).astype(jnp.float32)
        theta_dot_after = jnp.where(done, x_after_components[3], theta_dot_next).astype(jnp.float32)
        t_after = jnp.where(done, jnp.asarray(0, dtype=jnp.int32), t_next)

        new_state = {
            "x": x_after,
            "x_dot": x_dot_after,
            "theta": theta_after,
            "theta_dot": theta_dot_after,
            "t": t_after,
            "regime": regime_after,
        }
        obs = self._normalize_obs(
            new_state["x"], new_state["x_dot"],
            new_state["theta"], new_state["theta_dot"],
        )
        # Likelihood for the env-agnostic Belief wrapper. Computed from the
        # actual (theta, theta_dot, action) pair plus the *observed* next
        # angular velocity — exactly the data the wrapper needs to update
        # the analytical posterior. Done = True case is fine: the wrapper
        # zeros the belief on done regardless of likelihood.
        regime_likelihood = self._regime_likelihood(
            theta, theta_dot, action, theta_dot_next,
        )
        info = {
            "regime": regime,           # regime active during this step
            "regime_next": regime_next,
            "theta": theta_next,
            "theta_dot": theta_dot_next,
            "regime_likelihood": regime_likelihood,
        }
        return new_state, obs, reward, done, info
