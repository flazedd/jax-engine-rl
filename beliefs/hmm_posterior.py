"""Analytical forward algorithm for the MM env's HMM posterior.

Given the regime-switching env's transition matrix and per-regime fill
probabilities, filter the belief over regimes from observed actions and
fills. The update is exact (no approximation) and pure JAX so it can live
inside the rollout scan.

Semantics:
  filter_step(b, a, bid_fill, ask_fill, q, env) →
      P(r_t = r | evidence up to and including step t), length n_regimes.

  predict_step(b, env) →
      P(r_{t+1} = r' | same evidence), length n_regimes,
      computed as b_filtered @ T.

In practice, the env wrapper maintains `b` as "the belief at the start of
the current step" (i.e., already predicted forward), so the full update
per step is filter_step → predict_step.

Boundary handling. When inventory is at ±I_max, one side cannot fill;
the observed fill=0 on that side carries no regime information. The
likelihood uses the *effective* fill probability `p_raw * can_fill`, so
a blocked side contributes a neutral factor and the belief updates only
from the unblocked side. This matches the generative process exactly.
"""
from __future__ import annotations

import chex
import jax.numpy as jnp

from envs.mm_reduced import (
    _ASK_SPREADS,
    _BID_SPREADS,
    _TIGHT_MASK_ASK,
    _TIGHT_MASK_BID,
    MMReducedEnv,
)


def initial_belief(env: MMReducedEnv) -> chex.Array:
    return jnp.asarray(env.initial_distribution, dtype=jnp.float32)


def transition_matrix(env: MMReducedEnv) -> chex.Array:
    return jnp.asarray(env.transition_matrix, dtype=jnp.float32).reshape(
        env.n_regimes, env.n_regimes
    )


def per_regime_fill_probs(
    env: MMReducedEnv, action: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Return (p_bid, p_ask), each shape [n_regimes], for the given action.

    `action` is a scalar int. Output is per-regime raw fill probability.
    """
    bid_tight = _TIGHT_MASK_BID[action]
    ask_tight = _TIGHT_MASK_ASK[action]
    pt_bid = jnp.asarray(env.regime_p_tight_bid, dtype=jnp.float32)
    pw_bid = jnp.asarray(env.regime_p_wide_bid, dtype=jnp.float32)
    pt_ask = jnp.asarray(env.regime_p_tight_ask, dtype=jnp.float32)
    pw_ask = jnp.asarray(env.regime_p_wide_ask, dtype=jnp.float32)
    p_bid = jnp.where(bid_tight, pt_bid, pw_bid)
    p_ask = jnp.where(ask_tight, pt_ask, pw_ask)
    return p_bid, p_ask


def likelihood(
    env: MMReducedEnv,
    action: chex.Array,
    bid_fill: chex.Array,
    ask_fill: chex.Array,
    q: chex.Array,
) -> chex.Array:
    """P(bid_fill, ask_fill | regime, action, q) — shape [n_regimes].

    Accounts for inventory-bound rejection: if q == +I_max the bid cannot
    fill (so bid_fill is deterministically 0 regardless of regime) and the
    bid side contributes 1.0 to the likelihood.
    """
    p_bid_raw, p_ask_raw = per_regime_fill_probs(env, action)
    bid_ok = (q < env.inventory_max).astype(jnp.float32)
    ask_ok = (q > -env.inventory_max).astype(jnp.float32)
    p_bid_eff = p_bid_raw * bid_ok  # shape [n_regimes]
    p_ask_eff = p_ask_raw * ask_ok

    # Bernoulli likelihood. Guard for p==0 with bid_fill==1 (impossible) by
    # keeping floating arithmetic; that branch is masked by the generative
    # process so we never see it at runtime.
    bf = bid_fill.astype(jnp.float32)
    af = ask_fill.astype(jnp.float32)
    lik_bid = p_bid_eff * bf + (1.0 - p_bid_eff) * (1.0 - bf)
    lik_ask = p_ask_eff * af + (1.0 - p_ask_eff) * (1.0 - af)
    return lik_bid * lik_ask


def filter_step(
    belief: chex.Array,
    env: MMReducedEnv,
    action: chex.Array,
    bid_fill: chex.Array,
    ask_fill: chex.Array,
    q: chex.Array,
) -> chex.Array:
    """One filtering step: P(r_t | ev through t-1) → P(r_t | ev through t)."""
    lik = likelihood(env, action, bid_fill, ask_fill, q)
    post = belief * lik
    # Guard against an all-zero posterior (numerically possible under tiny
    # probabilities even if unreachable under the generative process).
    total = jnp.sum(post)
    post = jnp.where(total > 0, post / jnp.maximum(total, 1e-12), belief)
    return post


def predict_step(belief: chex.Array, env: MMReducedEnv) -> chex.Array:
    """Predict forward: b_{t+1}(r') = sum_r T(r → r') b_filtered(r)."""
    T = transition_matrix(env)
    return belief @ T


def full_update(
    belief: chex.Array,
    env: MMReducedEnv,
    action: chex.Array,
    bid_fill: chex.Array,
    ask_fill: chex.Array,
    q: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    """Filter then predict. Returns (b_filtered_t, b_pred_{t+1}).

    The filtered belief is the posterior on r_t given evidence through t;
    the predicted belief is the prior on r_{t+1}. The wrapper uses the
    predicted belief as the agent's next-step input.
    """
    b_filt = filter_step(belief, env, action, bid_fill, ask_fill, q)
    b_pred = predict_step(b_filt, env)
    return b_filt, b_pred


def entropy(belief: chex.Array) -> chex.Array:
    """Shannon entropy (nats)."""
    b = jnp.clip(belief, 1e-12, 1.0)
    return -jnp.sum(b * jnp.log(b), axis=-1)
