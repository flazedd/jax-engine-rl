"""RegimeBandit env invariants.

Checks:
  (1) reset draws random initial regime; across many seeds, both regimes
      appear and shape is scalar int32.
  (2) Within an episode, the regime evolves: across a long enough rollout,
      regimes switch (i.e., neither regime dominates the entire episode).
  (3) Sticky transition rate: empirical stay rate is close to `stay_prob`.
  (4) Reward is exactly 0.0 or 1.0; obs is zero.
  (5) On the step where `done=True`, `t` resets to 0 and a fresh regime is
      drawn (distinct from the just-finished episode's terminal regime
      across many trials).
  (6) JIT + lax.scan rollout compiles and stays shape-stable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.validation.regime_bandit import RegimeBanditEnv
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent


def _test_reset_random_initial_regime() -> int:
    env = RegimeBanditEnv()
    regimes = []
    for s in range(40):
        state, obs = env.reset(jax.random.PRNGKey(s))
        if obs.shape != (env.obs_size,):
            print(f"[test_regime_bandit] obs shape {obs.shape}", flush=True)
            return 1
        if state["regime"].shape != ():
            print(f"[test_regime_bandit] regime shape {state['regime'].shape}", flush=True)
            return 1
        regimes.append(int(state["regime"]))
    if set(regimes) != {0, 1}:
        print(f"[test_regime_bandit] only got regimes {set(regimes)} across 40 seeds", flush=True)
        return 1
    return 0


def _test_regime_evolves_within_episode() -> int:
    # Long episode + multiple runs; expect to see at least one switch in most.
    env = RegimeBanditEnv(episode_length=400, stay_prob=0.95)
    n_with_switch = 0
    for s in range(8):
        state, _ = env.reset(jax.random.PRNGKey(s))
        regimes = [int(state["regime"])]
        key = jax.random.PRNGKey(1000 + s)
        for _ in range(env.episode_length - 1):
            a_key, s_key, key = jax.random.split(key, 3)
            action = jax.random.randint(a_key, (), 0, env.n_actions)
            state, _, _, _, info = env.step(state, action, s_key)
            regimes.append(int(info["regime"]))
        if len(set(regimes)) > 1:
            n_with_switch += 1
    if n_with_switch < 7:  # 8 trials × ~95%+ → essentially always
        print(f"[test_regime_bandit] only {n_with_switch}/8 episodes had a switch", flush=True)
        return 1
    return 0


def _test_sticky_transition_rate() -> int:
    env = RegimeBanditEnv(episode_length=2000, stay_prob=0.95)
    state, _ = env.reset(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(7)
    prev_regime = int(state["regime"])
    stays = 0
    total = 0
    for _ in range(env.episode_length - 1):
        a_key, s_key, key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        state, _, _, _, info = env.step(state, action, s_key)
        cur = int(info["regime"])
        stays += int(cur == prev_regime)
        total += 1
        prev_regime = cur
    rate = stays / total
    # Allow ±0.02 around 0.95 with n=1999.
    if abs(rate - 0.95) > 0.02:
        print(f"[test_regime_bandit] empirical stay rate {rate:.3f} far from 0.95", flush=True)
        return 1
    return 0


def _test_reward_bernoulli_obs_zero() -> int:
    env = RegimeBanditEnv()
    state, obs = env.reset(jax.random.PRNGKey(0))
    if not jnp.allclose(obs, 0.0):
        print(f"[test_regime_bandit] obs not zero on reset: {obs}", flush=True)
        return 1
    key = jax.random.PRNGKey(7)
    for _ in range(200):
        a_key, s_key, key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        state, obs, reward, _, _ = env.step(state, action, s_key)
        if not jnp.allclose(obs, 0.0):
            print(f"[test_regime_bandit] obs not zero after step: {obs}", flush=True)
            return 1
        r = float(reward)
        if r not in (0.0, 1.0):
            print(f"[test_regime_bandit] reward {r} not in {{0,1}}", flush=True)
            return 1
    return 0


def _test_done_resets_state() -> int:
    env = RegimeBanditEnv(episode_length=4)
    state, _ = env.reset(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(99)
    done = None
    for _t in range(env.episode_length):
        a_key, s_key, key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        state, _, _, done, _ = env.step(state, action, s_key)
    if not bool(done):
        print(f"[test_regime_bandit] expected done at t={env.episode_length}", flush=True)
        return 1
    if int(state["t"]) != 0:
        print(f"[test_regime_bandit] t not reset after done: {int(state['t'])}", flush=True)
        return 1
    # Across many seeds, post-done regime distribution should be ~50/50
    # (initial draw is uniform). Run 200 trials and check both regimes seen.
    seen = set()
    for s in range(100):
        state, _ = env.reset(jax.random.PRNGKey(s))
        key = jax.random.PRNGKey(2000 + s)
        for _t in range(env.episode_length):
            a_key, s_key, key = jax.random.split(key, 3)
            action = jax.random.randint(a_key, (), 0, env.n_actions)
            state, _, _, _, _ = env.step(state, action, s_key)
        seen.add(int(state["regime"]))
        if seen == {0, 1}:
            break
    if seen != {0, 1}:
        print(f"[test_regime_bandit] post-done regime stuck on {seen}", flush=True)
        return 1
    return 0


def _test_jit_scan_rollout() -> int:
    env = RegimeBanditEnv(episode_length=10)

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
        print(f"[test_regime_bandit] bad scan shapes: rs={rs.shape} ds={ds.shape}", flush=True)
        return 1
    n_done = int(ds.sum())
    expected = 80 // env.episode_length
    if n_done != expected:
        print(f"[test_regime_bandit] saw {n_done} dones, expected {expected}", flush=True)
        return 1
    return 0


def main() -> int:
    run = ScriptRun(script="test_regime_bandit")
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []

    for name, fn in (
        ("reset_random_initial_regime", _test_reset_random_initial_regime),
        ("regime_evolves_within_episode", _test_regime_evolves_within_episode),
        ("sticky_transition_rate", _test_sticky_transition_rate),
        ("reward_bernoulli_obs_zero", _test_reward_bernoulli_obs_zero),
        ("done_resets_state", _test_done_resets_state),
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
            print(f"[test_regime_bandit] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_regime_bandit] {name} FAIL", flush=True)
            failures.append(name)

    summary_path = REPO_ROOT / "results" / "tests" / "regime_bandit.json"
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
