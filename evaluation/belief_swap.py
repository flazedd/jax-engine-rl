"""Counterfactual belief-swap diagnostic.

The locked-regime action distribution measures what a policy *did*. It has two
known blind spots: a policy that responds to the regime by steering into
different inventories rather than by acting differently at the same inventory
is scored as unresponsive, and the inventory levels where the comparison is
possible at all shrink to whatever the policy visits in every regime.

This diagnostic measures the belief-to-action map instead. The observation is
held fixed and only the belief input is varied, so the question becomes: given
the same situation, does this policy act differently when it believes it is in
a different regime? Coverage is chosen rather than observed, so every method is
scored on identical inputs, and steering never enters.

The null is exact rather than empirical. A policy with no belief input, the
regime-agnostic agent, cannot respond to a swapped belief and scores zero by
construction.

Belief substitution per method family:
  - RL²      : the incoming GRU carry, which is the belief the policy reads
               after folding in the observation.
  - VariBAD  : the variational mean and log-variance handed to the policy.
  - Belief / Oracle PPO : the trailing `n_regimes` entries of the observation.
  - Regime-agnostic / stacked-obs : no belief input, so the score is zero.

Belief spaces differ across method families, so absolute values compare only
within a method. The concat-versus-hypernet comparison, which is the decisive
one, is within-method by construction.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

N_ACTIONS_DEFAULT = 3


# ---------------------------------------------------------------------------
# Action distribution under a substituted belief
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def action_probs_with_belief(
    agent: Any,
    agent_state: Any,
    obs: np.ndarray,       # [B, obs_dim] observations to hold fixed
    beliefs: np.ndarray,   # [B, belief_dim] beliefs to substitute
    family: str,
    log_vars: np.ndarray | None = None,  # [B, latent_dim], VariBAD only
    belief_slice: tuple[int, int] | None = None,  # obs_belief family only
) -> np.ndarray:
    """Return P(action | obs, substituted belief) as [B, n_actions].

    The policy parameters are untouched; only the belief input changes.
    """
    obs_j = jnp.asarray(obs, dtype=jnp.float32)
    bel_j = jnp.asarray(beliefs, dtype=jnp.float32)

    if family == "rl2":
        # The policy reads the carry produced from (incoming carry, obs).
        # Substituting the incoming carry is the belief swap: the GRU folds in
        # the same observation either way, exactly as it does when acting.
        model = agent._model()
        params = {"params": agent_state["params"]["params"]}

        def one(carry_i, obs_i):
            _new_carry, logits, _value, _simplex = model.apply(params, carry_i, obs_i)
            return logits

        logits = jax.vmap(one)(bel_j, obs_j)

    elif family == "varibad":
        if log_vars is None:
            raise ValueError("VariBAD belief swap needs the matching log_var")
        policy = agent._policy()
        params = agent_state["params"]["policy"]
        lv_j = jnp.asarray(log_vars, dtype=jnp.float32)

        def one(obs_i, mu_i, lv_i):
            logits, _value, _simplex = policy.apply(params, obs_i, mu_i, lv_i)
            return logits

        logits = jax.vmap(one)(obs_j, bel_j, lv_j)

    elif family == "obs_belief":
        # Belief-PPO and Oracle-PPO carry the regime block inside the
        # observation. Its position is NOT trailing: the augmented tuple wraps
        # the belief wrapper, so the layout is
        # [inventory, belief, prev_action, prev_reward, done] and the block has
        # to be located explicitly rather than taken off the end.
        if belief_slice is None:
            raise ValueError("obs_belief family needs an explicit belief_slice")
        lo, hi = belief_slice
        if hi - lo != bel_j.shape[-1]:
            raise ValueError(
                f"belief_slice {belief_slice} spans {hi - lo} entries but the "
                f"substituted belief has {bel_j.shape[-1]}"
            )
        swapped = jnp.concatenate(
            [obs_j[:, :lo], bel_j, obs_j[:, hi:]], axis=-1,
        )
        model = agent._model()
        params = agent_state["params"]

        def one(obs_i):
            logits, _value = model.apply(params, obs_i)
            return logits

        logits = jax.vmap(one)(swapped)

    else:
        raise ValueError(f"unknown belief-swap family: {family!r}")

    return _softmax(np.asarray(logits, dtype=np.float64))


# ---------------------------------------------------------------------------
# Separation score
# ---------------------------------------------------------------------------


def _max_pairwise_tv(probs_by_regime: np.ndarray) -> float:
    """Largest total-variation distance between two regimes' action
    distributions. `probs_by_regime` is [n_regimes, n_actions]."""
    n = probs_by_regime.shape[0]
    return max(
        0.5 * float(np.abs(probs_by_regime[i] - probs_by_regime[j]).sum())
        for i in range(n)
        for j in range(i + 1, n)
    )


def belief_swap_separation(
    agent: Any,
    agent_state: Any,
    family: str,
    obs_by_inventory: list[np.ndarray],     # per inventory: [M, obs_dim]
    beliefs_by_regime: list[list[np.ndarray]],   # [regime][inventory] -> [K, dim]
    log_vars_by_regime: list[list[np.ndarray]] | None = None,
    rng: np.random.Generator | None = None,
    n_samples: int = 256,
    hist_by_regime: list[list[np.ndarray]] | None = None,
    n_inventory_dims: int | None = None,
    hold_belief_fixed: bool = False,
    belief_slice: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Mean over inventory levels of the across-regime separation.

    At each inventory level the observations are real ones recorded at that
    level, pooled across regimes so the situation is regime-neutral. The *same*
    sampled observations are then paired with beliefs drawn from each regime in
    turn, so the only thing that differs between the three action distributions
    being compared is the belief.

    Beliefs are drawn from the *same* inventory level as the observation they
    are paired with. A method's belief is correlated with its inventory, so
    pooling beliefs across inventories would query the policy at belief and
    inventory combinations it never encounters, and would dilute precisely the
    methods whose belief and inventory are most tightly coupled.
    """
    n_regimes = len(beliefs_by_regime)
    if family == "none":
        # No belief input: a swapped belief cannot change the action, so the
        # score is zero by construction rather than by measurement.
        return {
            "separation": 0.0,
            "per_inventory_separation": [0.0] * len(obs_by_inventory),
            "exact_null": True,
        }

    rng = rng or np.random.default_rng(0)
    per_inv: list[float] = []
    covered: list[int] = []

    for qi, obs_pool in enumerate(obs_by_inventory):
        pools = [beliefs_by_regime[r][qi] for r in range(n_regimes)]
        # Every regime must have supplied beliefs at this inventory, otherwise
        # there is nothing to compare against there.
        if len(obs_pool) == 0 or any(len(p) == 0 for p in pools):
            per_inv.append(float("nan"))
            covered.append(0)
            continue
        k = min(n_samples, len(obs_pool))
        obs_idx = rng.choice(len(obs_pool), size=k, replace=len(obs_pool) < k)
        obs_batch = obs_pool[obs_idx]

        # Holding the belief fixed isolates the observation-history channel:
        # every regime is scored with the *same* regime-neutral beliefs, pooled
        # across regimes, so only the history block differs between conditions.
        if hold_belief_fixed:
            fixed_pool = np.concatenate(pools)
            fixed_idx = rng.choice(
                len(fixed_pool), size=k, replace=len(fixed_pool) < k,
            )
            fixed_lv = (
                np.concatenate([log_vars_by_regime[r][qi] for r in range(n_regimes)])[
                    fixed_idx
                ]
                if (family == "varibad" and log_vars_by_regime is not None)
                else None
            )

        probs = []
        for r in range(n_regimes):
            pool = pools[r]
            b_idx = rng.choice(len(pool), size=k, replace=len(pool) < k)
            lv = (
                log_vars_by_regime[r][qi][b_idx]
                if (family == "varibad" and log_vars_by_regime is not None)
                else None
            )
            if hold_belief_fixed:
                pool, b_idx, lv = fixed_pool, fixed_idx, fixed_lv
            obs_r = obs_batch
            if hist_by_regime is not None and n_inventory_dims is not None:
                # The RL²-style observation carries last-step action, reward and
                # fills, which is regime evidence in its own right. Swapping it
                # alongside the belief measures the policy's response to all the
                # regime information it receives, not just the accumulated part.
                h_pool = hist_by_regime[r][qi]
                h_idx = rng.choice(len(h_pool), size=k, replace=len(h_pool) < k)
                obs_r = np.concatenate(
                    [obs_batch[:, :n_inventory_dims], h_pool[h_idx]], axis=-1,
                )
            p = action_probs_with_belief(
                agent, agent_state, obs_r, pool[b_idx], family, log_vars=lv,
                belief_slice=belief_slice,
            )
            probs.append(p.mean(axis=0))
        per_inv.append(_max_pairwise_tv(np.stack(probs)))
        covered.append(int(min(len(p) for p in pools)))

    vals = np.asarray(per_inv, dtype=np.float64)
    ok = ~np.isnan(vals)
    return {
        "separation": float(vals[ok].mean()) if ok.any() else float("nan"),
        "per_inventory_separation": [
            None if np.isnan(v) else float(v) for v in vals
        ],
        "inventory_pool_sizes": covered,
        "exact_null": False,
    }
