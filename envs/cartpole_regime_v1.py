"""Regime-switching CartPole — second POMDP env for the decoupling thesis.

A continuous-state cartpole with HMM-switching action-success
probabilities. The agent balances a pole on a cart; each step it
chooses to push left (action 0) or right (action 1) at a fixed
force magnitude (10 N). The latent regime governs the per-direction
*success probability* — in some regimes leftward pushes are
reliable but rightward pushes mostly fail, in others the reverse.
A failed push delivers zero force.

Why asymmetric stochastic actions. Earlier attempts with regime-
conditioned gravity, force-magnitude, and wind force all failed R1
(policy divergence) because cartpole's state observation
(x, ẋ, θ, θ̇) is itself a sufficient statistic for control: state-
feedback PPO infers the relevant disturbance within a few steps and
reacts optimally without needing explicit regime info. Stochastic
actions break this by making the *direction* of the optimal action
distribution genuinely regime-dependent — in a regime where right
pushes mostly fail, the optimal policy must bias toward left even
when pole tilt would normally call for right. This mirrors the M5
MarketMakingV1 structure where regime determines which action class
(sym / favor-ask / favor-bid) is optimal, and the regime-agnostic
policy is forced to a compromise that bleeds reward.

The analytical posterior is a closed-form Bayesian filter. The
likelihood under each regime hypothesis is a mixture of two
Gaussians: P(obs | regime) = p_succeed · N(obs; predicted-applied)
+ (1 − p_succeed) · N(obs; predicted-no-force), capturing the
two possible underlying force events under that regime's success
probability for the executed action.

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
_FORCE_MAG: float = 10.0       # N — fixed agent push magnitude per step
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
    # Per-(regime, action) probability that the agent's push delivers
    # force this step. Row-major flat 1D tuple of length
    # n_regimes * n_actions = 3 * 2 = 6. Index r * n_actions + a.
    # Default: r0 favours left pushes, r1 symmetric, r2 favours right.
    regime_action_success: tuple = (
        0.95, 0.30,
        0.80, 0.80,
        0.30, 0.95,
    )
    # Row-major flat 1D tuple of length n_regimes ** 2.
    transition_matrix: tuple = field(default_factory=tuple)
    initial_distribution: tuple = field(default_factory=tuple)
    # -1 → regime transitions freely; >=0 → regime locked at that index.
    lock_regime: int = -1

    def __post_init__(self) -> None:
        for name in ("regime_action_success", "transition_matrix", "initial_distribution"):
            v = getattr(self, name)
            if not isinstance(v, tuple):
                object.__setattr__(self, name, tuple(v))
        n = self.n_regimes
        if len(self.regime_action_success) != n * N_ACTIONS:
            raise ValueError(
                f"regime_action_success must have length {n * N_ACTIONS} "
                f"(n_regimes × n_actions), got {len(self.regime_action_success)}"
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

    def _success_probabilities_per_regime(
        self, action: chex.Array
    ) -> chex.Array:
        """P(action succeeds | regime), shape [n_regimes]."""
        mat = jnp.asarray(self.regime_action_success, dtype=jnp.float32).reshape(
            self.n_regimes, N_ACTIONS,
        )
        return mat[:, action.astype(jnp.int32)]

    def _regime_likelihood(
        self,
        theta: chex.Array,
        theta_dot: chex.Array,
        action: chex.Array,
        observed_theta_dot_next: chex.Array,
    ) -> chex.Array:
        """P(observed θ-dot transition | regime, prev state, action) — [n_regimes].

        Mixture of two Gaussians per regime: with probability p_r,a
        the agent's push of magnitude ±F_MAG is applied this step;
        with probability 1−p_r,a no force acts. We compute the
        Gaussian density of the observed θ-dot under each event,
        weight by the regime-conditional success probability, and
        sum. Returned as un-normalised probabilities (the Bayesian
        filter normalises them; constant prefactors cancel). Log-
        space max-shift for numerical stability.
        """
        f_action = jnp.where(action == 1, _FORCE_MAG, -_FORCE_MAG)
        theta_ddot_succeed = self._angular_acceleration(theta, theta_dot, f_action)
        theta_ddot_fail = self._angular_acceleration(
            theta, theta_dot, jnp.asarray(0.0, dtype=jnp.float32),
        )
        predicted_succeed = theta_dot + theta_ddot_succeed * _DT
        predicted_fail = theta_dot + theta_ddot_fail * _DT
        sigma = self.angular_velocity_noise_std
        log_density_succeed = -0.5 * ((observed_theta_dot_next - predicted_succeed) / sigma) ** 2
        log_density_fail = -0.5 * ((observed_theta_dot_next - predicted_fail) / sigma) ** 2

        p_succ = self._success_probabilities_per_regime(action)
        # log_lik(r) = logsumexp(log p_r + log_density_succeed,
        #                       log(1 − p_r) + log_density_fail).
        # Shape ops keep [n_regimes].
        eps = 1e-12
        log_p = jnp.log(jnp.clip(p_succ, eps, 1.0))
        log_one_minus_p = jnp.log(jnp.clip(1.0 - p_succ, eps, 1.0))
        a = log_p + log_density_succeed
        b = log_one_minus_p + log_density_fail
        m = jnp.maximum(a, b)
        log_lik = m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m))
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
        # Determine whether the agent's push succeeds this step under the
        # true regime's success probability for the chosen direction.
        success_mat = jnp.asarray(
            self.regime_action_success, dtype=jnp.float32
        ).reshape(self.n_regimes, N_ACTIONS)
        p_success_now = success_mat[regime, action]
        k_success, k_noise, k_reg, k_init = jax.random.split(key, 4)
        u = jax.random.uniform(k_success, ())
        succeeded = u < p_success_now
        f_action = jnp.where(action == 1, _FORCE_MAG, -_FORCE_MAG)
        force = jnp.where(succeeded, f_action, jnp.asarray(0.0, dtype=jnp.float32))

        # Forward Euler integration under the (possibly stochastically
        # zeroed) action force.
        theta = state["theta"]
        theta_dot = state["theta_dot"]
        theta_ddot = self._angular_acceleration(theta, theta_dot, force)
        x_ddot = (
            (force + _POLE_MASS_LENGTH * theta_dot**2 * jnp.sin(theta))
            / _TOTAL_MASS
            - _POLE_MASS_LENGTH * theta_ddot * jnp.cos(theta) / _TOTAL_MASS
        )

        # Process noise on angular velocity. This keeps the posterior
        # non-degenerate even given the action-success Bernoulli signal.
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
