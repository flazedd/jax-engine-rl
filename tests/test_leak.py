"""Regime-information-leak test.

Ensures that regime-agnostic observations (plain MarketMakingV1 and BeliefObsEnv
at initial-belief time) do NOT contain the true regime. Belief obs is allowed
to *correlate* with regime, but at reset time the belief is the HMM prior and
cannot depend on the sampled regime.

Concretely:
  - Plain MarketMakingV1: reset obs must be identical across regimes (it's a
    one-hot over inventory only).
  - BeliefObsEnv: reset obs depends only on initial inventory and the fixed
    `initial_distribution`; again identical across regimes at t=0.
  - OracleObsEnv: reset obs DOES depend on regime (that's its job). Positive
    control for the test.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.market_making_v1 import MarketMakingV1
from envs.wrappers.belief_obs import BeliefObsEnv
from envs.wrappers.oracle_obs import OracleObsEnv
from training.config import _load_yaml_with_extends
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
E_FINAL = REPO_ROOT / "experiments" / "configs" / "envs" / "e_final.yaml"


def _env_params() -> dict:
    return _load_yaml_with_extends(E_FINAL)["env"]["params"]


def _reset_obs(env, lock_regime: int) -> np.ndarray:
    """Return the initial observation when regime is locked to lock_regime."""
    if hasattr(env, "inner"):
        inner = env.inner
    else:
        inner = env
    # Build a locked env, pass through wrapper if needed.
    locked_inner = MarketMakingV1(**{**_env_params(), "lock_regime": lock_regime})
    if isinstance(env, OracleObsEnv):
        locked = OracleObsEnv(inner=locked_inner)
    elif isinstance(env, BeliefObsEnv):
        locked = BeliefObsEnv(inner=locked_inner)
    else:
        locked = locked_inner
    _, obs = locked.reset(jax.random.PRNGKey(0))
    return np.asarray(obs)


def _test_plain_env_no_leak() -> int:
    """Reset obs of plain env is regime-invariant."""
    env = MarketMakingV1(**_env_params())
    obs_by_r = [_reset_obs(env, r) for r in range(env.n_regimes)]
    ref = obs_by_r[0]
    ok = all(np.allclose(o, ref) for o in obs_by_r)
    return 0 if ok else 1


def _test_belief_env_no_leak_at_t0() -> int:
    """Reset obs of belief env is regime-invariant (belief = prior)."""
    env = BeliefObsEnv(inner=MarketMakingV1(**_env_params()))
    obs_by_r = [_reset_obs(env, r) for r in range(env.inner.n_regimes)]
    ref = obs_by_r[0]
    ok = all(np.allclose(o, ref) for o in obs_by_r)
    return 0 if ok else 1


def _test_oracle_env_leaks_by_design() -> int:
    """Positive control: OracleObsEnv obs DOES change with regime."""
    env = OracleObsEnv(inner=MarketMakingV1(**_env_params()))
    n = env.inner.n_regimes
    obs_by_r = [_reset_obs(env, r) for r in range(n)]
    # All pairwise obs must differ in the regime tail.
    base_size = env.inner.obs_size
    tails = [o[base_size:] for o in obs_by_r]
    # Each tail is a one-hot — the argmax must equal the regime id.
    ok = all(int(np.argmax(t)) == r for r, t in enumerate(tails))
    return 0 if ok else 1


def main() -> int:
    run = ScriptRun(script="test_leak")
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []

    for name, fn in (
        ("plain_env_no_leak", _test_plain_env_no_leak),
        ("belief_env_no_leak_at_t0", _test_belief_env_no_leak_at_t0),
        ("oracle_env_leaks_by_design", _test_oracle_env_leaks_by_design),
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
            print(f"[test_leak] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_leak] {name} FAIL", flush=True)
            failures.append(name)

    summary_path = REPO_ROOT / "results" / "tests" / "leak.json"
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
