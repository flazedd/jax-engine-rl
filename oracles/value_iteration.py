"""Value iteration on the full-info MDP.

Full-info MDP: state = (inventory, regime). The optimal policy on this MDP
is what Oracle-PPO and per-regime PPO target. Used in M2 for:
- R1 (policy divergence): compare per-regime argmax policies.
- R2 (locked-regime optimality): compare PPO return to VI return on each
  locked regime.
- R3 / R4 utility: approximate episode-return under a given policy by
  forward-evaluating the MDP for `episode_length` steps from q=0, regime
  drawn from the initial distribution.

Implementation: numpy Bellman iteration on the analytically-enumerated
(P, R) tables — same pattern as `oracles/analytical_as.py`, generalized
to a regime dimension.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.mm_reduced import (
    ACTION_FAVOR_ASK,
    ACTION_FAVOR_BID,
    ACTION_SYM,
    MMReducedEnv,
    N_ACTIONS,
)


@dataclass(frozen=True)
class VIResult:
    """Optimal per-state value and policy on the full-info (inv, regime) MDP."""

    # V[inv, regime] — optimal value on the full-info state.
    V: np.ndarray
    # policy[inv, regime] — argmax action.
    policy: np.ndarray
    # Q[inv, regime, action] — action-value function.
    Q: np.ndarray
    # Expected undiscounted per-step reward under optimal policy, per regime,
    # computed by forward rollout from q=0 with the regime locked to r.
    per_regime_expected_step_reward: np.ndarray
    # Expected episode return = per_step * episode_length, per regime.
    per_regime_expected_episode_return: np.ndarray
    # Expected episode return under the mixed policy (regime sampled from
    # initial distribution at each reset).
    mixed_expected_episode_return: float


def _fill_probs_np(env: MMReducedEnv, action: int, regime: int) -> tuple[float, float]:
    bid_tight = action in (ACTION_SYM, ACTION_FAVOR_BID)
    ask_tight = action in (ACTION_SYM, ACTION_FAVOR_ASK)
    if env.n_regimes == 1:
        p_bid = env.p_tight if bid_tight else env.p_wide
        p_ask = env.p_tight if ask_tight else env.p_wide
    else:
        p_bid = (
            env.regime_p_tight_bid[regime] if bid_tight else env.regime_p_wide_bid[regime]
        )
        p_ask = (
            env.regime_p_tight_ask[regime] if ask_tight else env.regime_p_wide_ask[regime]
        )
    return float(p_bid), float(p_ask)


def _bid_ask_spreads(action: int) -> tuple[float, float]:
    if action == ACTION_SYM:
        return 1.0, 1.0
    if action == ACTION_FAVOR_ASK:
        return 3.0, 1.0
    if action == ACTION_FAVOR_BID:
        return 1.0, 3.0
    raise ValueError(action)


def _transition_tables(
    env: MMReducedEnv,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (P, R) for the full-info MDP.

    P[s, r, a, s', r'] = transition prob.
    R[s, r, a]        = expected one-step reward.

    Dimensions: s, s' over n_inventory_states; r, r' over n_regimes.
    """
    n_inv = env.n_inventory_states
    n_reg = max(1, env.n_regimes)
    kappa = env.inventory_penalty

    # Transition over regime.
    if n_reg == 1:
        T = np.ones((1, 1), dtype=np.float64)
    else:
        T = np.asarray(env.transition_matrix, dtype=np.float64).reshape(n_reg, n_reg)

    P = np.zeros((n_inv, n_reg, N_ACTIONS, n_inv, n_reg), dtype=np.float64)
    R = np.zeros((n_inv, n_reg, N_ACTIONS), dtype=np.float64)

    for s in range(n_inv):
        q = s - env.inventory_max
        bid_ok = q < env.inventory_max
        ask_ok = q > -env.inventory_max
        for r in range(n_reg):
            for a in range(N_ACTIONS):
                p_bid, p_ask = _fill_probs_np(env, a, r)
                bid_spread, ask_spread = _bid_ask_spreads(a)

                # Enumerate Bernoulli outcomes.
                for bf_raw in (0, 1):
                    pb = p_bid if bf_raw == 1 else (1.0 - p_bid)
                    bf_eff = bf_raw if bid_ok else 0
                    for af_raw in (0, 1):
                        pa = p_ask if af_raw == 1 else (1.0 - p_ask)
                        af_eff = af_raw if ask_ok else 0
                        prob_fill = pb * pa
                        q_next = q + bf_eff - af_eff
                        capture = bf_eff * bid_spread + af_eff * ask_spread
                        inv_pen = kappa * (q_next ** 2)
                        reward = capture - inv_pen
                        s_next = q_next + env.inventory_max
                        # Regime transitions independently.
                        for r_next in range(n_reg):
                            prob = prob_fill * T[r, r_next]
                            P[s, r, a, s_next, r_next] += prob
                        R[s, r, a] += prob_fill * reward

    return P, R


def _bellman_iterate(
    P: np.ndarray, R: np.ndarray, gamma: float, tol: float = 1e-9, max_iter: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    n_inv, n_reg, n_act, _, _ = P.shape
    V = np.zeros((n_inv, n_reg), dtype=np.float64)
    # Reshape P to [n_inv, n_reg, n_act, n_inv * n_reg] for einsum convenience.
    P_flat = P.reshape(n_inv, n_reg, n_act, n_inv * n_reg)
    for _ in range(max_iter):
        V_flat = V.reshape(n_inv * n_reg)
        Q = R + gamma * P_flat @ V_flat  # [n_inv, n_reg, n_act]
        V_new = Q.max(axis=-1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    V_flat = V.reshape(n_inv * n_reg)
    Q = R + gamma * P_flat @ V_flat
    return V, Q


def _stationary_distribution(P_pi: np.ndarray) -> np.ndarray:
    n = P_pi.shape[0]
    A = P_pi.T - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi_d, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi_d = np.clip(pi_d, 0.0, None)
    total = pi_d.sum()
    if total <= 0:
        return np.ones(n) / n
    return pi_d / total


def _per_step_reward_under_policy(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    regime_lock: int | None,
) -> float:
    """Expected undiscounted per-step reward under `policy`.

    If `regime_lock` is given, restrict to that regime (per-regime evaluation).
    Otherwise average over the full (inv, regime) stationary distribution.
    """
    n_inv, n_reg, n_act, _, _ = P.shape
    # Build induced transition matrix on the chosen state space.
    if regime_lock is None:
        # Full (inv × regime) chain.
        n_s = n_inv * n_reg
        P_pi = np.zeros((n_s, n_s), dtype=np.float64)
        r_pi = np.zeros(n_s, dtype=np.float64)
        for s in range(n_inv):
            for r in range(n_reg):
                a = int(policy[s, r])
                row = P[s, r, a].reshape(n_s)
                idx = s * n_reg + r
                P_pi[idx] = row
                r_pi[idx] = R[s, r, a]
        pi_d = _stationary_distribution(P_pi)
        return float(np.dot(pi_d, r_pi))

    # Restrict to single regime: transitions projected into (s, r_lock).
    r = regime_lock
    n_s = n_inv
    P_proj = np.zeros((n_s, n_s), dtype=np.float64)
    r_proj = np.zeros(n_s, dtype=np.float64)
    for s in range(n_inv):
        a = int(policy[s, r])
        # Sum over r' → s' conditioning on staying in regime r (since locked).
        # Equivalent to marginalizing the regime transition out: treat regime
        # as frozen by setting regime dynamics to identity. We instead take
        # the inventory-only transition from the action dynamics.
        # P[s, r, a, s', r']: sum over r' to get inventory marginal.
        inv_row = P[s, r, a].sum(axis=1)  # [n_inv]
        P_proj[s] = inv_row
        r_proj[s] = R[s, r, a]
    pi_d = _stationary_distribution(P_proj)
    return float(np.dot(pi_d, r_proj))


def solve_value_iteration(env: MMReducedEnv) -> VIResult:
    """Run VI on the full-info (inv, regime) MDP for the given env config."""
    P, R = _transition_tables(env)
    V, Q = _bellman_iterate(P, R, env.gamma)
    policy = np.argmax(Q, axis=-1).astype(np.int64)
    n_reg = max(1, env.n_regimes)

    per_regime_step = np.zeros(n_reg, dtype=np.float64)
    for r in range(n_reg):
        per_regime_step[r] = _per_step_reward_under_policy(P, R, policy, r)
    per_regime_ep = per_regime_step * env.episode_length

    mixed_step = _per_step_reward_under_policy(P, R, policy, None)
    mixed_ep = mixed_step * env.episode_length

    return VIResult(
        V=V,
        policy=policy,
        Q=Q,
        per_regime_expected_step_reward=per_regime_step,
        per_regime_expected_episode_return=per_regime_ep,
        mixed_expected_episode_return=float(mixed_ep),
    )


def compromise_policy_expected_returns(
    env: MMReducedEnv, vi: VIResult | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute the expected return of the best *regime-agnostic* policy.

    The compromise policy selects one action per inventory (regardless of
    regime) to maximize *marginalized* return under the HMM's stationary
    regime distribution. This is the VI-derivable ceiling for what a
    regime-agnostic agent can do; used as a sanity check on R3 (a
    regime-agnostic PPO should approach this, not Oracle-PPO).

    Returns (per_regime_return, mixed_return, compromise_policy[inv]).
    """
    if vi is None:
        vi = solve_value_iteration(env)
    P, R = _transition_tables(env)
    n_inv = env.n_inventory_states
    n_reg = max(1, env.n_regimes)

    if n_reg == 1:
        # Compromise = VI policy (they coincide).
        compromise = vi.policy[:, 0].astype(np.int64)
    else:
        # Use stationary regime distribution as the weight.
        T = np.asarray(env.transition_matrix, dtype=np.float64).reshape(n_reg, n_reg)
        w = _stationary_distribution(T)
        # Q[s, r, a] weighted over regimes → Q_marg[s, a].
        Q_marg = np.einsum("sra,r->sa", vi.Q, w)
        compromise = np.argmax(Q_marg, axis=-1).astype(np.int64)

    # Evaluate this policy's per-regime and mixed return.
    # Build policy[s, r] = compromise[s] (regime-agnostic).
    per_regime_policy = np.broadcast_to(compromise[:, None], (n_inv, n_reg)).astype(
        np.int64
    )
    per_regime_step = np.zeros(n_reg, dtype=np.float64)
    for r in range(n_reg):
        per_regime_step[r] = _per_step_reward_under_policy(
            P, R, per_regime_policy, r
        )
    per_regime_ep = per_regime_step * env.episode_length
    mixed_step = _per_step_reward_under_policy(P, R, per_regime_policy, None)
    mixed_ep = mixed_step * env.episode_length
    return per_regime_ep, float(mixed_ep), compromise


def policy_disagreement(vi: VIResult) -> tuple[float, np.ndarray]:
    """R1 metric: fraction of inventory states where per-regime policies disagree.

    Disagreement at state s: not all regimes pick the same action at s.
    Returns (fraction, per_state_disagrees bool array).
    """
    policy = vi.policy  # [n_inv, n_regime]
    first = policy[:, 0:1]
    disagrees = np.any(policy != first, axis=1)
    return float(disagrees.mean()), disagrees


def _policy_values_on_locked_regime(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    regime_lock: int,
    gamma: float,
) -> np.ndarray:
    """Exact V^π(s) on the regime_lock-locked inventory chain.

    Uses policy evaluation via linear solve on the inventory marginal:
        V = R_π + γ P_π V   ⇒   V = (I − γ P_π)^{-1} R_π.
    """
    n_inv = P.shape[0]
    P_proj = np.zeros((n_inv, n_inv), dtype=np.float64)
    r_proj = np.zeros(n_inv, dtype=np.float64)
    for s in range(n_inv):
        a = int(policy[s, regime_lock])
        P_proj[s] = P[s, regime_lock, a].sum(axis=1)
        r_proj[s] = R[s, regime_lock, a]
    A = np.eye(n_inv) - gamma * P_proj
    return np.linalg.solve(A, r_proj)


def wrong_regime_value_loss(
    env: MMReducedEnv, vi: VIResult
) -> tuple[float, float, np.ndarray]:
    """R1 metric: value loss from COMMITTING to the wrong regime's policy.

    For each (r_true, r_other != r_true) we forward-roll the MDP under
    π_{r_other} on the locked r_true dynamics, giving a per-step reward
    w(r_other → r_true). Compare to the locked optimum v(r_true):

        abs_loss   = mean over pairs of max(v(r_true) − w(·→r_true), 0)
        rel_loss   = mean over pairs of that loss / v(r_true)

    This is in per-step-reward units (same as `per_regime_expected_step_reward`),
    so rel_loss is a proper fraction of attainable value — unlike the
    V − Q one-step gap, which under γ→1 is dominated by one-step cost even
    when policies diverge badly.

    `per_state_rel_loss` is the forward-rolled *relative* loss per
    (r_true, r_other, inventory), flattened — policy-commitment value-loss
    fraction under locked r_true dynamics. Large values mean committing to
    π_{r_other} at state s in r_true is costly over the full horizon.

    Returns (abs_loss, rel_loss, per_state_rel_loss[flat]).
    """
    policy = vi.policy
    n_inv, n_reg = vi.V.shape

    # Forward-rolled loss in per-step-reward units (scalar, unchanged).
    if n_reg <= 1:
        return 0.0, 0.0, np.zeros(0, dtype=np.float64)
    P, R_tab = _transition_tables(env)
    per_regime_opt = vi.per_regime_expected_step_reward
    abs_losses: list[float] = []
    rel_losses: list[float] = []
    per_state_rel: list[float] = []
    for r_true in range(n_reg):
        v_true = _policy_values_on_locked_regime(
            P, R_tab, policy, r_true, env.gamma
        )
        denom = np.maximum(np.abs(v_true), 1e-6)
        for r_other in range(n_reg):
            if r_other == r_true:
                continue
            pol = policy.copy()
            pol[:, r_true] = policy[:, r_other]
            wrong_step = _per_step_reward_under_policy(P, R_tab, pol, r_true)
            opt_step = float(per_regime_opt[r_true])
            loss = max(opt_step - wrong_step, 0.0)
            abs_losses.append(loss)
            if opt_step > 1e-9:
                rel_losses.append(loss / opt_step)

            v_other = _policy_values_on_locked_regime(
                P, R_tab, pol, r_true, env.gamma
            )
            per_s = np.clip((v_true - v_other) / denom, 0.0, None)
            per_state_rel.extend(per_s.tolist())
    abs_mean = float(np.mean(abs_losses)) if abs_losses else 0.0
    rel_mean = float(np.mean(rel_losses)) if rel_losses else 0.0
    return abs_mean, rel_mean, np.asarray(per_state_rel, dtype=np.float64)
