"""Phase 5 tests — performance benchmarks."""
import pytest
import jax
import jax.numpy as jnp
import time

from lob_sim.config import SimConfig
from lob_sim.step import run_episode

T = 5000
config = SimConfig(max_steps=T)


class TestPerformance:
    def test_single_env_throughput(self):
        """Warm single-env throughput > 10,000 steps/sec."""
        # Compile
        run_jit = jax.jit(lambda k, a: run_episode(config, k, a, locked_regime=-1))
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        key = jax.random.PRNGKey(0)
        run_jit(key, actions)  # compile
        jax.block_until_ready(run_jit(key, actions))  # ensure ready
        # Time
        t0 = time.perf_counter()
        _, out = run_jit(key, actions)
        jax.block_until_ready(out["mid_price"])
        elapsed = time.perf_counter() - t0
        steps_per_sec = T / elapsed
        print(f"Single env: {steps_per_sec:.0f} steps/sec")
        assert steps_per_sec > 10_000

    def test_vmap_compiles(self):
        """vmap over 64 envs compiles without error."""
        batch_size = 64
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        batched = jax.jit(jax.vmap(
            lambda k: run_episode(config, k, actions, locked_regime=-1)
        ))
        _, out = batched(keys)
        jax.block_until_ready(out["mid_price"])
        assert out["mid_price"].shape == (batch_size, T)

    def test_vmap_scaling(self):
        """Throughput scales with batch size."""
        results = {}
        actions = jnp.full((T,), 4, dtype=jnp.int32)
        for batch_size in [1, 16, 64, 256]:
            keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
            batched = jax.jit(jax.vmap(
                lambda k: run_episode(config, k, actions, locked_regime=-1)
            ))
            batched(keys)  # compile
            jax.block_until_ready(batched(keys))
            t0 = time.perf_counter()
            _, out = batched(keys)
            jax.block_until_ready(out["mid_price"])
            elapsed = time.perf_counter() - t0
            total_steps = batch_size * T
            results[batch_size] = total_steps / elapsed
            print(f"Batch {batch_size}: {results[batch_size]:.0f} total steps/sec")
        # Batch 64 should be faster total throughput than batch 1
        assert results[64] > results[1] * 2

    def test_no_oom_batch_1024(self):
        """Batch size 1024 doesn't OOM (may be slow, just verify it works)."""
        keys = jax.random.split(jax.random.PRNGKey(0), 1024)
        actions = jnp.full((1000,), 4, dtype=jnp.int32)  # shorter T
        config_short = SimConfig(max_steps=1000)
        batched = jax.jit(jax.vmap(
            lambda k: run_episode(config_short, k, actions, locked_regime=-1)
        ))
        _, out = batched(keys)
        jax.block_until_ready(out["mid_price"])
        assert out["mid_price"].shape == (1024, 1000)
