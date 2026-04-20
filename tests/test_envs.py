"""Env invariants for the regime-switching MM env.

Checks:
  (1) lock_regime holds the regime constant across a full rollout.
  (2) Stationary regime frequencies over a long rollout match the HMM's
      stationary distribution within 5%.
  (3) Inventory stays within [-I_max, +I_max] at every step.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.mm_reduced import MMReducedEnv
from training.config import _load_yaml_with_extends
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
E_FINAL = REPO_ROOT / "experiments" / "configs" / "envs" / "e_final.yaml"


def _env_params() -> dict:
    return _load_yaml_with_extends(E_FINAL)["env"]["params"]


def _simulate(env: MMReducedEnv, n_steps: int, seed: int = 0):
    """Random-policy rollout collecting (q, regime) per step. Shape [T]."""

    def body(carry, _):
        state, key = carry
        a_key, s_key, next_key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        new_state, _obs, _r, _d, info = env.step(state, action, s_key)
        return (new_state, next_key), (new_state["q"], new_state["regime"])

    key = jax.random.PRNGKey(seed)
    reset_key, key = jax.random.split(key)
    state, _ = env.reset(reset_key)
    _, (qs, regimes) = jax.lax.scan(body, (state, key), xs=None, length=n_steps)
    return np.asarray(qs), np.asarray(regimes)


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    n = T.shape[0]
    b = np.ones(n) / n
    for _ in range(5000):
        b = b @ T
    return b


def _test_lock_regime_holds() -> int:
    params = _env_params()
    ok = True
    for r in range(params["n_regimes"]):
        locked = MMReducedEnv(**{**params, "lock_regime": r})
        # Need a long-enough episode, but env.episode_length caps scan len above —
        # use env's episode_length.
        _, regimes = _simulate(locked, locked.episode_length, seed=r)
        if not np.all(regimes == r):
            ok = False
            print(
                f"[test_envs] lock_regime={r} broken: got unique {np.unique(regimes)}",
                flush=True,
            )
    return 0 if ok else 1


def _test_stationary_distribution_matches() -> int:
    params = _env_params()
    env = MMReducedEnv(**params)
    n = env.n_regimes
    T = np.asarray(params["transition_matrix"], dtype=np.float64).reshape(n, n)
    stationary = _stationary_distribution(T)
    # Long rollout — 200 episodes' worth of steps. The 0.98 diagonal means
    # the chain has ~50-step correlation length, so effective sample size is
    # ~500 even at n_steps=25k. Threshold 0.03 is still comfortably above the
    # expected sampling noise.
    n_steps = 200 * env.episode_length
    _, regimes = _simulate(env, n_steps, seed=123)
    freqs = np.bincount(regimes, minlength=n) / regimes.size
    err = np.max(np.abs(freqs - stationary))
    print(
        f"[test_envs] stationary freqs={freqs} target={stationary} max_err={err:.3f}",
        flush=True,
    )
    return 0 if err < 0.03 else 1


def _test_inventory_bounds() -> int:
    params = _env_params()
    env = MMReducedEnv(**params)
    qs, _ = _simulate(env, 20 * env.episode_length, seed=7)
    if qs.min() < -env.inventory_max or qs.max() > env.inventory_max:
        print(
            f"[test_envs] inventory out of bounds: min={qs.min()} max={qs.max()}",
            flush=True,
        )
        return 1
    return 0


def main() -> int:
    run = ScriptRun(script="test_envs")
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []

    for name, fn in (
        ("lock_regime_holds", _test_lock_regime_holds),
        ("stationary_matches", _test_stationary_distribution_matches),
        ("inventory_bounds", _test_inventory_bounds),
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
            print(f"[test_envs] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_envs] {name} FAIL", flush=True)
            failures.append(name)

    summary_path = REPO_ROOT / "results" / "tests" / "envs.json"
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
