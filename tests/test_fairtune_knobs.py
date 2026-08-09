"""Unit tests for the M5R fair-tune knobs added to agents.

Verifies that the two new tunable parameters wire through:
  - belief_layernorm  (concat-side architectural-care knob)
  - hypernet_init_scale (hypernet-side init scale, 0.0 = Beck zero-init)

Tests are forward-pass only (no training); they confirm:
  1. Defaults preserve previous behaviour bit-for-bit on the same key.
  2. belief_layernorm=True changes the concat-path output.
  3. hypernet_init_scale=0.0 yields uniform logits at init (zero-init).
  4. Configs splat into the agent dataclass without errors.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.rl2 import GRUActorCritic, RL2Agent
from agents.varibad import VariBADAgent, VariBADPolicy


KEY = jax.random.PRNGKey(0)
OBS_SIZE = 11
N_ACTIONS = 3
HIDDEN = 16
LATENT = 4


def _rl2_forward(model: GRUActorCritic, key) -> jnp.ndarray:
    carry = jnp.zeros((HIDDEN,), dtype=jnp.float32)
    obs = jnp.ones((OBS_SIZE,), dtype=jnp.float32)
    params = model.init(key, carry, obs)
    _, logits, _, _ = model.apply(params, carry, obs)
    return logits


def test_rl2_concat_default_unchanged():
    """Default belief_layernorm=False must match a model with no LN at all."""
    base = GRUActorCritic(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, hidden_dim=HIDDEN,
        integration="concat",
    )
    new = GRUActorCritic(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, hidden_dim=HIDDEN,
        integration="concat",
        belief_layernorm=False,
        hypernet_init_scale=0.01,
    )
    a = _rl2_forward(base, KEY)
    b = _rl2_forward(new, KEY)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_rl2_concat_layernorm_changes_output():
    """LayerNorm on the GRU hidden must change the concat logits."""
    no_ln = GRUActorCritic(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, hidden_dim=HIDDEN,
        integration="concat", belief_layernorm=False,
    )
    with_ln = GRUActorCritic(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, hidden_dim=HIDDEN,
        integration="concat", belief_layernorm=True,
    )
    a = _rl2_forward(no_ln, KEY)
    b = _rl2_forward(with_ln, KEY)
    assert not np.allclose(a, b, atol=1e-4), (
        "belief_layernorm=True should change the concat-path output"
    )


def test_rl2_hypernet_zero_init_uniform_logits():
    """Zero-init hypernet output → all flat_weights = 0 → logits = 0."""
    model = GRUActorCritic(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, hidden_dim=HIDDEN,
        integration="hypernet",
        hypernet_target_hidden=8, hypernet_hidden=16,
        hypernet_init_scale=0.0,
    )
    logits = _rl2_forward(model, KEY)
    # With zero kernel and zero bias on hypernet output, all target weights
    # are zero → logits should be exactly zero.
    np.testing.assert_allclose(logits, jnp.zeros_like(logits), atol=1e-6)


def test_rl2_dataclass_accepts_new_kwargs():
    """RL2Agent dataclass accepts the two new YAML kwargs."""
    agent = RL2Agent(
        obs_size=OBS_SIZE, n_actions=N_ACTIONS,
        belief_layernorm=True, hypernet_init_scale=0.001,
    )
    assert agent.belief_layernorm is True
    assert agent.hypernet_init_scale == 0.001


# --- VariBAD --------------------------------------------------------------


def _varibad_policy_forward(policy: VariBADPolicy, key) -> jnp.ndarray:
    obs = jnp.ones((OBS_SIZE,), dtype=jnp.float32)
    mu = jnp.ones((LATENT,), dtype=jnp.float32) * 0.5
    log_var = jnp.zeros((LATENT,), dtype=jnp.float32)
    params = policy.init(key, obs, mu, log_var)
    logits, _, _ = policy.apply(params, obs, mu, log_var)
    return logits


def test_varibad_concat_default_unchanged():
    base = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="concat",
    )
    new = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="concat",
        belief_layernorm=False, hypernet_init_scale=0.01,
    )
    a = _varibad_policy_forward(base, KEY)
    b = _varibad_policy_forward(new, KEY)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_varibad_concat_layernorm_changes_output():
    no_ln = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="concat", belief_layernorm=False,
    )
    with_ln = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="concat", belief_layernorm=True,
    )
    a = _varibad_policy_forward(no_ln, KEY)
    b = _varibad_policy_forward(with_ln, KEY)
    assert not np.allclose(a, b, atol=1e-4), (
        "belief_layernorm=True should change the VariBAD concat-path output"
    )


def test_varibad_hypernet_zero_init_uniform_logits():
    policy = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="hypernet",
        hypernet_target_hidden=8, hypernet_hidden=16,
        hypernet_init_scale=0.0,
    )
    logits = _varibad_policy_forward(policy, KEY)
    np.testing.assert_allclose(logits, jnp.zeros_like(logits), atol=1e-6)


def test_varibad_dataclass_accepts_new_kwargs():
    agent = VariBADAgent(
        obs_size=OBS_SIZE, n_actions=N_ACTIONS,
        belief_layernorm=True, hypernet_init_scale=0.001,
    )
    assert agent.belief_layernorm is True
    assert agent.hypernet_init_scale == 0.001


def test_varibad_hypernet_layernorm_is_noop():
    """belief_layernorm should be ignored in the hypernet path (no LN applied)."""
    no_ln = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="hypernet",
        hypernet_target_hidden=8, hypernet_hidden=16,
        belief_layernorm=False,
    )
    with_ln = VariBADPolicy(
        n_actions=N_ACTIONS, obs_size=OBS_SIZE, latent_dim=LATENT,
        hidden_dim=HIDDEN, integration="hypernet",
        hypernet_target_hidden=8, hypernet_hidden=16,
        belief_layernorm=True,
    )
    a = _varibad_policy_forward(no_ln, KEY)
    b = _varibad_policy_forward(with_ln, KEY)
    np.testing.assert_allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    test_rl2_concat_default_unchanged()
    test_rl2_concat_layernorm_changes_output()
    test_rl2_hypernet_zero_init_uniform_logits()
    test_rl2_dataclass_accepts_new_kwargs()
    test_varibad_concat_default_unchanged()
    test_varibad_concat_layernorm_changes_output()
    test_varibad_hypernet_zero_init_uniform_logits()
    test_varibad_dataclass_accepts_new_kwargs()
    test_varibad_hypernet_layernorm_is_noop()
    print("[test_fairtune_knobs] OK | tests=9 | output=tests/test_fairtune_knobs.py")
