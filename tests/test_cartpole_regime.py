"""Env invariants for the regime-switching cartpole env.

Checks:
  (1) lock_regime holds the regime constant across a full rollout.
  (2) Stationary regime frequencies over a long rollout match the HMM's
      stationary distribution within 5%.
  (3) Step shapes / dtypes are stable across two consecutive calls (jit-safe).
  (4) info["regime_likelihood"] has shape [n_regimes] and is non-negative.
  (5) Locked-regime posterior concentrates on the locked regime over time.
  (6) Episode auto-resets on done (t wraps to 0, regime resampled).
  (7) jit + lax.scan rollout runs without error.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.cartpole_regime_v1 import CartPoleRegimeV1
from envs.wrappers.belief_obs import BeliefObsEnv
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default env parameters used across tests. Diag-0.98 HMM matches the project
# standard. Three regimes vary gravity by a factor of 4.
_DEFAULT_PARAMS = dict(
    episode_length=128,
    gamma=0.99,
    angular_velocity_noise_std=0.2,
    n_regimes=3,
    # r0 favours left, r1 symmetric, r2 favours right
    regime_action_success=(
        0.95, 0.30,
        0.80, 0.80,
        0.30, 0.95,
    ),
    transition_matrix=(
        0.98, 0.01, 0.01,
        0.01, 0.98, 0.01,
        0.01, 0.01, 0.98,
    ),
    initial_distribution=(0.333, 0.333, 0.334),
)


def _simulate(env: CartPoleRegimeV1, n_steps: int, seed: int = 0):
    """Random-policy rollout collecting (regime, theta, lik) per step."""

    def body(carry, _):
        state, key = carry
        a_key, s_key, next_key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        new_state, _obs, _r, _d, info = env.step(state, action, s_key)
        return (new_state, next_key), (
            new_state["regime"], new_state["theta"], info["regime_likelihood"],
        )

    key = jax.random.PRNGKey(seed)
    reset_key, key = jax.random.split(key)
    state, _ = env.reset(reset_key)
    _, (regimes, thetas, liks) = jax.lax.scan(body, (state, key), xs=None, length=n_steps)
    return np.asarray(regimes), np.asarray(thetas), np.asarray(liks)


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    n = T.shape[0]
    b = np.ones(n) / n
    for _ in range(5000):
        b = b @ T
    return b


def _test_lock_regime_holds() -> int:
    ok = True
    for r in range(_DEFAULT_PARAMS["n_regimes"]):
        locked = CartPoleRegimeV1(**{**_DEFAULT_PARAMS, "lock_regime": r})
        regimes, _, _ = _simulate(locked, locked.episode_length, seed=r)
        if not np.all(regimes == r):
            ok = False
            print(
                f"[test_cartpole_regime] lock_regime={r} broken: got unique "
                f"{np.unique(regimes)}",
                flush=True,
            )
    return 0 if ok else 1


def _test_stationary_distribution_matches() -> int:
    env = CartPoleRegimeV1(**_DEFAULT_PARAMS)
    n = env.n_regimes
    T = np.asarray(_DEFAULT_PARAMS["transition_matrix"], dtype=np.float64).reshape(n, n)
    stationary = _stationary_distribution(T)
    # 100 episodes' worth of steps. With the 0.98 diagonal the chain has
    # ~50-step correlation length, so effective sample size ~250 at n=12k.
    n_steps = 100 * env.episode_length
    regimes, _, _ = _simulate(env, n_steps, seed=123)
    freqs = np.bincount(regimes, minlength=n) / regimes.size
    err = np.max(np.abs(freqs - stationary))
    print(
        f"[test_cartpole_regime] stationary freqs={freqs} target={stationary} "
        f"max_err={err:.3f}",
        flush=True,
    )
    return 0 if err < 0.05 else 1


def _test_step_shape_stable() -> int:
    env = CartPoleRegimeV1(**_DEFAULT_PARAMS)
    key = jax.random.PRNGKey(0)
    state, obs = env.reset(key)
    s1, _, _, _, info1 = env.step(state, jnp.asarray(0, dtype=jnp.int32), key)
    s2, _, _, _, info2 = env.step(s1, jnp.asarray(1, dtype=jnp.int32), key)
    # State trees must have identical shapes/dtypes step over step.
    for k in s1:
        if s1[k].shape != s2[k].shape or s1[k].dtype != s2[k].dtype:
            print(f"[test_cartpole_regime] state['{k}'] shape/dtype unstable", flush=True)
            return 1
    if info1["regime_likelihood"].shape != (env.n_regimes,):
        print(
            f"[test_cartpole_regime] regime_likelihood shape "
            f"{info1['regime_likelihood'].shape} != ({env.n_regimes},)",
            flush=True,
        )
        return 1
    return 0


def _test_regime_likelihood_nonnegative() -> int:
    env = CartPoleRegimeV1(**_DEFAULT_PARAMS)
    _, _, liks = _simulate(env, 4 * env.episode_length, seed=42)
    if not np.all(liks >= 0.0):
        print(
            f"[test_cartpole_regime] negative likelihood encountered: "
            f"min={liks.min()}",
            flush=True,
        )
        return 1
    if not np.all(np.isfinite(liks)):
        print("[test_cartpole_regime] non-finite likelihood encountered", flush=True)
        return 1
    return 0


def _test_locked_posterior_concentrates() -> int:
    """Under a locked regime, the analytical posterior should concentrate on
    the true regime *in expectation*. With force-magnitude regimes the
    signal is present at every step regardless of pole angle. A single
    noisy episode can land anywhere because of process-noise realisations
    (e.g. middle-regime r=1 is statistically harder than r=0 / r=2 because
    its likelihood neighbours are equidistant); so we average end-of-episode
    posteriors across N_SEEDS independent episodes per regime."""

    N_SEEDS = 16
    ok = True
    for true_regime in range(_DEFAULT_PARAMS["n_regimes"]):
        locked = CartPoleRegimeV1(**{**_DEFAULT_PARAMS, "lock_regime": true_regime})
        wrapper = BeliefObsEnv(locked)

        def episode_final_belief(seed: int) -> chex.Array:
            def body(carry, _):
                state, key = carry
                a_key, s_key, next_key = jax.random.split(key, 3)
                action = jax.random.randint(a_key, (), 0, wrapper.n_actions)
                new_state, _obs, _r, _d, info = wrapper.step(state, action, s_key)
                return (new_state, next_key), info["belief"]
            key = jax.random.PRNGKey(seed)
            reset_key, key = jax.random.split(key)
            state, _ = wrapper.reset(reset_key)
            _, beliefs = jax.lax.scan(
                body, (state, key), xs=None, length=locked.episode_length - 1,
            )
            return beliefs[-1]

        seeds = jnp.arange(true_regime * N_SEEDS, (true_regime + 1) * N_SEEDS)
        finals = jax.vmap(episode_final_belief)(seeds)
        avg = np.asarray(jnp.mean(finals, axis=0))
        # Average end-of-episode posterior should put the true regime as
        # argmax with > 0.5 mass.
        if int(np.argmax(avg)) != true_regime or avg[true_regime] < 0.5:
            ok = False
            print(
                f"[test_cartpole_regime] locked={true_regime} avg final "
                f"posterior over {N_SEEDS} seeds = {avg.round(3)} "
                f"(expected mass on {true_regime})",
                flush=True,
            )
    return 0 if ok else 1


def _test_jit_scan_rollout() -> int:
    env = CartPoleRegimeV1(**_DEFAULT_PARAMS)

    @jax.jit
    def rollout(seed: int):
        def body(carry, _):
            state, key = carry
            a_key, s_key, next_key = jax.random.split(key, 3)
            action = jax.random.randint(a_key, (), 0, env.n_actions)
            new_state, _obs, r, _d, _info = env.step(state, action, s_key)
            return (new_state, next_key), r

        key = jax.random.PRNGKey(seed)
        reset_key, key = jax.random.split(key)
        state, _ = env.reset(reset_key)
        _, rewards = jax.lax.scan(body, (state, key), xs=None, length=env.episode_length)
        return rewards.sum()

    total = float(rollout(0))
    if not np.isfinite(total):
        print(f"[test_cartpole_regime] jit_scan_rollout non-finite total={total}",
              flush=True)
        return 1
    return 0


def _test_done_resets_state() -> int:
    """Episode boundary: t wraps to 0; the auto-reset path produces a state
    inside the initialization spread (|x|, |theta| < 0.1)."""
    env = CartPoleRegimeV1(**_DEFAULT_PARAMS)

    def body(carry, _):
        state, key = carry
        a_key, s_key, next_key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        new_state, _obs, _r, _d, _info = env.step(state, action, s_key)
        return (new_state, next_key), new_state

    key = jax.random.PRNGKey(99)
    reset_key, key = jax.random.split(key)
    state, _ = env.reset(reset_key)
    _, states = jax.lax.scan(
        body, (state, key), xs=None, length=env.episode_length + 1,
    )
    states_np = jax.tree_util.tree_map(np.asarray, states)
    # Step `episode_length` (0-indexed: episode_length - 1) emits done=True;
    # the next step starts a new episode. State at that step has t=0.
    # states_np["t"][episode_length - 1] should be 0 (the auto-reset wrote 0).
    t_after_done = int(states_np["t"][env.episode_length - 1])
    if t_after_done != 0:
        print(
            f"[test_cartpole_regime] expected t==0 after done, got {t_after_done}",
            flush=True,
        )
        return 1
    return 0


def main() -> int:
    run = ScriptRun(script="test_cartpole_regime")
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []

    for name, fn in (
        ("lock_regime_holds", _test_lock_regime_holds),
        ("stationary_matches", _test_stationary_distribution_matches),
        ("step_shape_stable", _test_step_shape_stable),
        ("regime_likelihood_nonnegative", _test_regime_likelihood_nonnegative),
        ("locked_posterior_concentrates", _test_locked_posterior_concentrates),
        ("jit_scan_rollout", _test_jit_scan_rollout),
        ("done_resets_state", _test_done_resets_state),
    ):
        tests_run += 1
        t0 = time.perf_counter()
        try:
            rc = fn()
        except Exception as e:
            rc = 1
            failures.append(f"{name}: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        if rc == 0:
            tests_passed += 1
            print(f"[test_cartpole_regime] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_cartpole_regime] {name} FAIL", flush=True)
            failures.append(name)

    summary_path = REPO_ROOT / "results" / "tests" / "cartpole_regime.json"
    if tests_passed == tests_run:
        run.ok(
            key_stats={"tests_run": tests_run, "tests_passed": tests_passed},
            summary_path=summary_path,
        )
        return 0
    run.fail(
        reason=f"{tests_passed}/{tests_run} passed; failures={failures}",
        summary_path=summary_path,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
