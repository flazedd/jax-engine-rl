"""StackObsEnv invariants.

Checks:
  (1) obs_size = inner.obs_size * k; obs is a flat 1D vector.
  (2) After reset, all K slots are copies of the initial inner obs.
  (3) After a non-terminal step, buffer slides: slot[-1] == new inner obs;
      slots[:-1] == previous buffer's slots[1:].
  (4) On the step where done=True, buffer is filled with K copies of the
      next-episode initial obs (no leak from previous episode).
  (5) JIT + lax.scan rollout compiles and stays shape-stable across done
      boundaries.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.validation.gridworld import GridworldEnv
from envs.wrappers.stack_obs import StackObsEnv
from utils.script_output import ScriptRun

REPO_ROOT = Path(__file__).resolve().parent.parent


def _test_obs_size_and_shape() -> int:
    inner = GridworldEnv()
    env = StackObsEnv(inner=inner, k=3)
    if env.obs_size != inner.obs_size * 3:
        print(f"[test_stack_obs] obs_size {env.obs_size} != {inner.obs_size}*3", flush=True)
        return 1
    state, obs = env.reset(jax.random.PRNGKey(0))
    if obs.ndim != 1:
        print(f"[test_stack_obs] obs.ndim={obs.ndim}, want 1", flush=True)
        return 1
    if obs.shape[0] != env.obs_size:
        print(f"[test_stack_obs] obs.shape[0]={obs.shape[0]}, want {env.obs_size}", flush=True)
        return 1
    return 0


def _test_reset_fills_buffer() -> int:
    inner = GridworldEnv()
    env = StackObsEnv(inner=inner, k=4)
    state, obs = env.reset(jax.random.PRNGKey(7))
    buf = np.asarray(state["buffer"])
    if buf.shape[0] != 4:
        print(f"[test_stack_obs] buffer rows {buf.shape[0]} != 4", flush=True)
        return 1
    for i in range(4):
        if not np.allclose(buf[i], buf[0]):
            print(f"[test_stack_obs] buffer slot {i} != slot 0 after reset", flush=True)
            return 1
    return 0


def _test_step_slides_buffer() -> int:
    inner = GridworldEnv(episode_length=64)
    env = StackObsEnv(inner=inner, k=3)
    state, _ = env.reset(jax.random.PRNGKey(0))
    prev_buf = np.asarray(state["buffer"])
    new_state, _obs, _r, done, _ = env.step(
        state, jnp.asarray(0, dtype=jnp.int32), jax.random.PRNGKey(1)
    )
    if bool(done):
        print("[test_stack_obs] unexpected done on first step (episode_length=64)", flush=True)
        return 1
    new_buf = np.asarray(new_state["buffer"])
    inner_obs = np.asarray(inner._obs(new_state["inner"]["xy"]))
    # slot[0] of new_buf should equal slot[1] of prev_buf
    if not np.allclose(new_buf[0], prev_buf[1]):
        print("[test_stack_obs] new_buf[0] != prev_buf[1]", flush=True)
        return 1
    # slot[1] of new_buf should equal slot[2] of prev_buf
    if not np.allclose(new_buf[1], prev_buf[2]):
        print("[test_stack_obs] new_buf[1] != prev_buf[2]", flush=True)
        return 1
    # slot[-1] should equal the new inner obs
    if not np.allclose(new_buf[-1], inner_obs):
        print(f"[test_stack_obs] new_buf[-1]={new_buf[-1]} != inner_obs={inner_obs}", flush=True)
        return 1
    return 0


def _test_done_refills_buffer() -> int:
    inner = GridworldEnv(episode_length=2)
    env = StackObsEnv(inner=inner, k=3)
    state, _ = env.reset(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(99)
    done = False
    last_state = state
    for _ in range(2):
        a_key, s_key, key = jax.random.split(key, 3)
        a = jax.random.randint(a_key, (), 0, env.n_actions)
        last_state, _obs, _r, done, _ = env.step(last_state, a, s_key)
    if not bool(done):
        print("[test_stack_obs] expected done after 2 steps with episode_length=2", flush=True)
        return 1
    buf = np.asarray(last_state["buffer"])
    # All K slots should be identical (next episode's initial obs).
    for i in range(buf.shape[0]):
        if not np.allclose(buf[i], buf[0]):
            print(f"[test_stack_obs] buffer slot {i} != slot 0 after done", flush=True)
            return 1
    return 0


def _test_jit_scan_rollout() -> int:
    inner = GridworldEnv(episode_length=10)
    env = StackObsEnv(inner=inner, k=4)

    def body(carry, _):
        state, key = carry
        a_key, s_key, next_key = jax.random.split(key, 3)
        action = jax.random.randint(a_key, (), 0, env.n_actions)
        new_state, obs, r, d, _ = env.step(state, action, s_key)
        return (new_state, next_key), (obs, r, d)

    n_steps = 50

    @jax.jit
    def run(seed):
        key = jax.random.PRNGKey(seed)
        reset_key, key = jax.random.split(key)
        state, _ = env.reset(reset_key)
        _, (obs, rs, ds) = jax.lax.scan(body, (state, key), xs=None, length=n_steps)
        return obs, rs, ds

    obs, rs, ds = run(0)
    obs = np.asarray(obs)
    if obs.shape != (n_steps, env.obs_size):
        print(f"[test_stack_obs] obs scan shape {obs.shape} != ({n_steps}, {env.obs_size})", flush=True)
        return 1
    if int(np.asarray(ds).sum()) != n_steps // inner.episode_length:
        print(f"[test_stack_obs] dones {int(np.asarray(ds).sum())} != {n_steps//inner.episode_length}", flush=True)
        return 1
    return 0


def main() -> int:
    run = ScriptRun(script="test_stack_obs")
    tests_run = 0
    tests_passed = 0
    failures: list[str] = []
    for name, fn in (
        ("obs_size_and_shape", _test_obs_size_and_shape),
        ("reset_fills_buffer", _test_reset_fills_buffer),
        ("step_slides_buffer", _test_step_slides_buffer),
        ("done_refills_buffer", _test_done_refills_buffer),
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
            print(f"[test_stack_obs] {name} OK ({elapsed:.2f}s)", flush=True)
        else:
            print(f"[test_stack_obs] {name} FAIL", flush=True)
            failures.append(name)
    summary_path = REPO_ROOT / "results" / "tests" / "stack_obs.json"
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
