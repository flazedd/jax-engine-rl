"""Bandit env invariants.

Checks:
  (1) reset draws random arm probabilities (different keys → different probs).
  (2) Within an episode, arm probabilities stay constant across steps.
  (3) Reward is exactly 0.0 or 1.0; obs is a constant zero.
  (4) On the step where `done=True`, arm probabilities are resampled (the
      next state's probs differ from the just-finished episode's).
  (5) JIT + lax.scan rollout compiles and stays shape-stable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.validation.bandit import BanditEnv
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent


def _test_reset_random_arm_probs() -> int:
    env = BanditEnv()
    s0, _ = env.reset(jax.random.PRNGKey(0))
    s1, _ = env.reset(jax.random.PRNGKey(1))
    if jnp.allclose(s0["arm_probs"], s1["arm_probs"]):
        print("[test_bandit] arm_probs identical across reset keys", flush=True)
        return 1
    if s0["arm_probs"].shape != (env.n_arms,):
        print(f"[test_bandit] arm_probs shape {s0['arm_probs'].shape}", flush=True)
        return 1
    return 0


def _test_arm_probs_constant_within_episode() -> int:
    env = BanditEnv(episode_length=10)
    state, _ = env.reset(jax.random.PRNGKey(0))
    initial_probs = state["arm_probs"]
    key = jax.random.PRNGKey(42)
    for t in range(env.episode_length - 1):  # stop before done
        a_key, s_key, key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        state, _, _, done, _ = env.step(state, action, s_key)
        if not jnp.allclose(state["arm_probs"], initial_probs):
            print(f"[test_bandit] arm_probs drifted at step {t}", flush=True)
            return 1
        if bool(done):
            print(f"[test_bandit] unexpected done at step {t}", flush=True)
            return 1
    return 0


def _test_reward_is_bernoulli_and_obs_zero() -> int:
    env = BanditEnv(episode_length=200)
    state, obs = env.reset(jax.random.PRNGKey(0))
    if not jnp.allclose(obs, 0.0):
        print(f"[test_bandit] obs not zero on reset: {obs}", flush=True)
        return 1
    key = jax.random.PRNGKey(7)
    rewards: list[float] = []
    for _ in range(100):
        a_key, s_key, key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        state, obs, reward, _, _ = env.step(state, action, s_key)
        if not jnp.allclose(obs, 0.0):
            print(f"[test_bandit] obs not zero after step: {obs}", flush=True)
            return 1
        r = float(reward)
        if r not in (0.0, 1.0):
            print(f"[test_bandit] reward {r} not in {{0,1}}", flush=True)
            return 1
        rewards.append(r)
    return 0


def _test_arm_probs_resample_on_done() -> int:
    env = BanditEnv(episode_length=4)
    state, _ = env.reset(jax.random.PRNGKey(0))
    initial_probs = state["arm_probs"]
    key = jax.random.PRNGKey(99)
    for t in range(env.episode_length):
        a_key, s_key, key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        state, _, _, done, _ = env.step(state, action, s_key)
    if not bool(done):
        print(f"[test_bandit] expected done at t={env.episode_length}", flush=True)
        return 1
    if jnp.allclose(state["arm_probs"], initial_probs):
        print("[test_bandit] arm_probs unchanged after done — auto-reset broken", flush=True)
        return 1
    if int(state["t"]) != 0:
        print(f"[test_bandit] t not reset after done: {int(state['t'])}", flush=True)
        return 1
    return 0


def _test_jit_scan_rollout() -> int:
    env = BanditEnv(episode_length=10)

    def body(carry, _):
        state, key = carry
        a_key, s_key, next_key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        new_state, _obs, r, d, _ = env.step(state, action, s_key)
        return (new_state, next_key), (r, d)

    n_steps = 80

    @jax.jit
    def run(seed):
        key = jax.random.PRNGKey(seed)
        reset_key, key = jax.random.split(key)
        state, _ = env.reset(reset_key)
        _, (rs, ds) = jax.lax.scan(body, (state, key), xs=None, length=n_steps)
        return rs, ds

    rs, ds = run(0)
    rs = np.asarray(rs)
    ds = np.asarray(ds)
    if rs.shape != (80,) or ds.shape != (80,):
        print(f"[test_bandit] bad scan shapes: rs={rs.shape} ds={ds.shape}", flush=True)
        return 1
    n_done = int(ds.sum())
    expected = 80 // env.episode_length
    if n_done != expected:
        print(f"[test_bandit] saw {n_done} dones, expected {expected}", flush=True)
        return 1
    return 0


def main() -> int:
    run = ScriptRun(script="test_bandit")
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []

    for name, fn in (
        ("reset_random_arm_probs", _test_reset_random_arm_probs),
        ("arm_probs_constant_within_episode", _test_arm_probs_constant_within_episode),
        ("reward_is_bernoulli_and_obs_zero", _test_reward_is_bernoulli_and_obs_zero),
        ("arm_probs_resample_on_done", _test_arm_probs_resample_on_done),
        ("jit_scan_rollout", _test_jit_scan_rollout),
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
            print(f"[test_bandit] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_bandit] {name} FAIL", flush=True)
            failures.append(name)

    summary_path = REPO_ROOT / "results" / "tests" / "bandit.json"
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
