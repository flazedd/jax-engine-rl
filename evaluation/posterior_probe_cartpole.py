"""Posterior-quality probe for meta-RL methods on CartPoleRegimeV1.

Parallel module to `evaluation.posterior_probe` for the second-POMDP
external-validity probe. The two probes share `train_probe`,
`ProbeBundle`, and `load_experiment` — only the rollout collection +
analytical-posterior computation differ, because cartpole's likelihood
is a mixture of Gaussians on θ-dot residuals (computed inside the env
and exposed via `info["regime_likelihood"]`) rather than MM's Bernoulli
likelihood from `(bid_fill, ask_fill, q)`.

The cartpole analytical posterior path is actually *simpler* than MM's:
the env already exposes the per-step likelihood vector, so the wrapper
just needs to filter (multiply by likelihood, normalise) and predict
(multiply by transition matrix). This is exactly what `BeliefObsEnv`
does inline, so we replicate that here.
"""
from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np

from envs.cartpole_regime_v1 import CartPoleRegimeV1
from evaluation.posterior_probe import ProbeBundle, train_probe


def get_inner_cartpole(env: Any) -> CartPoleRegimeV1:
    """Walk down wrapper chain until we find the CartPoleRegimeV1."""
    cur = env
    while not isinstance(cur, CartPoleRegimeV1):
        if not hasattr(cur, "inner"):
            raise TypeError(
                f"could not find CartPoleRegimeV1 inside {type(cur).__name__} "
                "(no `inner` attribute)"
            )
        cur = cur.inner
    return cur


def collect_probe_rollouts_cartpole(
    env: Any,
    agent: Any,
    agent_state: chex.ArrayTree,
    n_rollouts: int,
    rollout_length: int,
    key: chex.PRNGKey,
) -> dict[str, np.ndarray]:
    """Run `n_rollouts` parallel rollouts, recording per-step:
      - belief: agent's internal belief (RL² carry, VariBAD μ)
      - regime: env's true regime
      - analytical_belief: filtered HMM posterior using info["regime_likelihood"]

    Returns dict shaped `[T, N, ...]` ready for `train_probe`.
    """
    inner_cp = get_inner_cartpole(env)
    n_regimes = inner_cp.n_regimes
    if n_regimes < 2:
        raise ValueError("probe requires n_regimes ≥ 2; got " + str(n_regimes))

    initial_distribution = jnp.asarray(
        inner_cp.initial_distribution, dtype=jnp.float32,
    )
    transition_T = jnp.asarray(
        inner_cp.transition_matrix, dtype=jnp.float32,
    ).reshape(n_regimes, n_regimes)

    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)

    reset_keys = jax.random.split(key, n_rollouts + 1)
    keys_init, key = reset_keys[:n_rollouts], reset_keys[-1]
    env_states, obses = reset_v(keys_init)

    carry = agent.init_carry(n_rollouts)
    belief_key = getattr(agent, "belief_key", None)
    if belief_key is None:
        raise ValueError("agent has no `belief_key` attribute")

    analytical_belief = jnp.broadcast_to(
        initial_distribution, (n_rollouts, n_regimes),
    )

    out_belief = []
    out_regime = []
    out_analytical = []

    @jax.jit
    def act_step(agent_state_, carry_, obs_, key_):
        keys = jax.random.split(key_, n_rollouts)
        return jax.vmap(
            lambda c, o, k: agent.act(agent_state_, c, o, k),
            in_axes=(0, 0, 0),
        )(carry_, obs_, keys)

    @jax.jit
    def hmm_step_v(b, regime_likelihood, done):
        """Filter + predict using `info["regime_likelihood"]` directly.

        Mirrors the inline math in `envs.wrappers.belief_obs.BeliefObsEnv.step`:
          b_filt(r)  ∝ b(r) · lik(r)              # Bayes update
          b_pred(r') = sum_r b_filt(r) · T(r→r')  # forward in time
        On `done`, reset to the env's initial distribution.

        Records `b_filt` (the posterior at the *current* step), since
        that is what the agent's belief vector at step t approximates.
        """
        def per_env(b_i, lik_i, d_i):
            post = b_i * lik_i
            total = jnp.sum(post)
            b_filt = jnp.where(total > 0, post / jnp.maximum(total, 1e-12), b_i)
            b_pred = b_filt @ transition_T
            b_init = initial_distribution
            b_next = jnp.where(d_i, b_init, b_pred)
            return b_filt, b_next
        return jax.vmap(per_env)(b, regime_likelihood, done)

    for t in range(rollout_length):
        carry_pre_act = carry

        step_key, key = jax.random.split(key)
        action, extras, new_carry = act_step(agent_state, carry, obses, step_key)

        if belief_key == "carry_in":
            belief_t = carry_pre_act
        else:
            if belief_key not in extras:
                raise KeyError(
                    f"belief_key {belief_key!r} not in agent.act extras "
                    f"(have: {list(extras.keys())})"
                )
            belief_t = extras[belief_key]

        step_keys = jax.random.split(key, n_rollouts + 1)
        env_keys, key = step_keys[:n_rollouts], step_keys[-1]
        env_states, obses_next, rewards, dones, info = step_v(
            env_states, action, env_keys,
        )

        b_filt_t, analytical_belief = hmm_step_v(
            analytical_belief, info["regime_likelihood"], dones,
        )

        out_belief.append(np.asarray(belief_t))
        out_regime.append(np.asarray(info["regime"]))
        out_analytical.append(np.asarray(b_filt_t))

        zeros = jnp.zeros_like(new_carry)
        mask = dones[:, None].astype(new_carry.dtype)
        carry = mask * zeros + (1.0 - mask) * new_carry
        obses = obses_next

    return {
        "belief": np.stack(out_belief),
        "regime": np.stack(out_regime).astype(np.int32),
        "analytical_belief": np.stack(out_analytical),
    }


def probe_one_seed_cartpole(
    bundle: ProbeBundle,
    n_rollouts: int = 200,
    rollout_length: int = 128,
    classifier: str = "logistic",
    test_frac: float = 0.2,
    rng_key: int = 0,
) -> dict[str, Any]:
    """Full cartpole probe pipeline for one (method, seed) checkpoint."""
    key = jax.random.PRNGKey(rng_key)
    data = collect_probe_rollouts_cartpole(
        bundle.env, bundle.agent, bundle.agent_state,
        n_rollouts=n_rollouts, rollout_length=rollout_length, key=key,
    )

    rng = np.random.default_rng(rng_key)
    perm = rng.permutation(n_rollouts)
    n_test = max(1, int(round(n_rollouts * test_frac)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    method_probe = train_probe(
        data["belief"], data["regime"], train_idx, test_idx,
        classifier=classifier, seed=rng_key,
        omega_TND=data["analytical_belief"],
    )
    analytical_probe = train_probe(
        data["analytical_belief"], data["regime"], train_idx, test_idx,
        classifier=classifier, seed=rng_key,
        omega_TND=data["analytical_belief"],
    )

    return {
        "method": method_probe,
        "analytical": analytical_probe,
        "n_rollouts": n_rollouts,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "rollout_length": rollout_length,
        "belief_dim": int(data["belief"].shape[-1]),
        "n_regimes_observed": int(data["regime"].max() + 1),
    }
