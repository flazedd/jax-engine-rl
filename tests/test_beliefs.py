"""Tests for the analytical HMM posterior.

Two checks:
  (1) filter_step agrees with a brute-force O(n) reference on random beliefs /
      random evidence. Used to catch indexing or normalization bugs.
  (2) predict_step collapses the stationary distribution back to itself — a
      direct property of the stationary left-eigenvector.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from beliefs.hmm_posterior import (
    filter_step,
    initial_belief,
    predict_step,
    transition_matrix,
)
from envs.mm_reduced import ACTION_FAVOR_ASK, ACTION_FAVOR_BID, ACTION_SYM, MMReducedEnv
from training.config import _load_yaml_with_extends
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent
E_FINAL = REPO_ROOT / "experiments" / "configs" / "envs" / "e_final.yaml"


def _load_env() -> MMReducedEnv:
    cfg = _load_yaml_with_extends(E_FINAL)["env"]
    return MMReducedEnv(**cfg["params"])


def _brute_filter(env: MMReducedEnv, b, action, bid_fill, ask_fill, q) -> np.ndarray:
    n = env.n_regimes
    post = np.zeros(n, dtype=np.float64)
    bid_ok = q < env.inventory_max
    ask_ok = q > -env.inventory_max
    bid_tight = action in (ACTION_SYM, ACTION_FAVOR_BID)
    ask_tight = action in (ACTION_SYM, ACTION_FAVOR_ASK)
    for r in range(n):
        p_bid = (
            env.regime_p_tight_bid[r] if bid_tight else env.regime_p_wide_bid[r]
        )
        p_ask = (
            env.regime_p_tight_ask[r] if ask_tight else env.regime_p_wide_ask[r]
        )
        # Effective probabilities (blocked side contributes neutral 1.0).
        pb_eff = p_bid if bid_ok else 0.0
        pa_eff = p_ask if ask_ok else 0.0
        lik_bid = pb_eff if bid_fill == 1 else (1.0 - pb_eff)
        lik_ask = pa_eff if ask_fill == 1 else (1.0 - pa_eff)
        post[r] = b[r] * lik_bid * lik_ask
    total = post.sum()
    if total <= 0:
        return np.asarray(b, dtype=np.float64)
    return post / total


def _test_filter_matches_brute(env: MMReducedEnv) -> int:
    """Iterate random (belief, action, fills, q) and check against brute force."""
    rng = np.random.default_rng(0)
    n_fail = 0
    n_total = 0
    for _ in range(200):
        alpha = rng.uniform(0.3, 3.0, env.n_regimes)
        b = rng.dirichlet(alpha)
        action = int(rng.integers(0, env.n_actions))
        q = int(rng.integers(-env.inventory_max, env.inventory_max + 1))
        bid_fill = int(rng.integers(0, 2))
        ask_fill = int(rng.integers(0, 2))
        # Respect the generative constraint: blocked-side fills can't happen.
        if q >= env.inventory_max:
            bid_fill = 0
        if q <= -env.inventory_max:
            ask_fill = 0

        ref = _brute_filter(env, b, action, bid_fill, ask_fill, q)
        out = np.asarray(
            filter_step(
                jnp.asarray(b, dtype=jnp.float32),
                env,
                jnp.asarray(action, dtype=jnp.int32),
                jnp.asarray(bid_fill, dtype=jnp.int32),
                jnp.asarray(ask_fill, dtype=jnp.int32),
                jnp.asarray(q, dtype=jnp.int32),
            )
        )
        n_total += 1
        if not np.allclose(out, ref, atol=1e-4):
            n_fail += 1
    return 0 if n_fail == 0 else 1


def _test_predict_on_stationary(env: MMReducedEnv) -> int:
    """Stationary regime distribution is a fixed point of predict_step."""
    T = np.asarray(transition_matrix(env))
    # Power-iterate to get stationary left-eigenvector.
    b = np.ones(env.n_regimes) / env.n_regimes
    for _ in range(2000):
        b = b @ T
    out = np.asarray(predict_step(jnp.asarray(b, dtype=jnp.float32), env))
    if np.allclose(out, b, atol=1e-4):
        return 0
    return 1


def _test_initial_belief_matches_config(env: MMReducedEnv) -> int:
    ref = np.asarray(env.initial_distribution, dtype=np.float64)
    out = np.asarray(initial_belief(env))
    if np.allclose(out, ref, atol=1e-6):
        return 0
    return 1


def main() -> int:
    run = ScriptRun(script="test_beliefs")
    env = _load_env()
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []

    for name, fn in (
        ("filter_matches_brute", _test_filter_matches_brute),
        ("predict_stationary_fixed_point", _test_predict_on_stationary),
        ("initial_belief_matches_config", _test_initial_belief_matches_config),
    ):
        tests_run += 1
        t0 = time.perf_counter()
        try:
            rc = fn(env)
        except Exception as e:
            rc = 1
            failures.append(f"{name}: {type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        if rc == 0:
            tests_passed += 1
            print(f"[test_beliefs] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_beliefs] {name} FAIL", flush=True)
            if not failures or not failures[-1].startswith(name):
                failures.append(name)

    summary_path = REPO_ROOT / "results" / "tests" / "beliefs.json"
    if tests_passed == tests_run:
        run.ok(
            key_stats={
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "env": E_FINAL.name,
            },
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
