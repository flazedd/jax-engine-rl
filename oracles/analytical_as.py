"""Analytical solution of the E0 reduced-form AS MDP.

"Analytical" here means the exact optimum of the finite MDP, obtained by
Bellman iteration on the analytically-constructed (transition, reward) tables.
It is not the Avellaneda & Stoikov (2008) continuous-time closed form — that
paper assumes continuous quote distances and Poisson arrivals. Our reduced
form uses 3 discrete quote configurations and independent Bernoulli fills per
side, for which Bellman iteration is the right notion of "analytical solution."

Outputs (`AnalyticalAS`):
  V            — optimal value per inventory state   (shape: [n_inventory])
  policy       — argmax action per inventory state   (shape: [n_inventory])
  action_probs — one-hot of `policy`                  (shape: [n_inventory, n_actions])
  skew         — P(favor_ask) − P(favor_bid) per inv (shape: [n_inventory])
  expected_episode_return
               — discounted return from q=0 under the optimal policy,
                 evaluated via forward rollout with the MDP's stochastic
                 dynamics. Used as the M1 ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.market_making_v1 import (
    ACTION_FAVOR_ASK,
    ACTION_FAVOR_BID,
    ACTION_SYM,
    MarketMakingV1,
    N_ACTIONS,
)


@dataclass(frozen=True)
class AnalyticalAS:
    V: np.ndarray
    policy: np.ndarray
    action_probs: np.ndarray
    skew: np.ndarray
    expected_episode_return: float
    q_margin: np.ndarray  # Q*(s, a*) − Q*(s, 2nd-best), per state. Zero ≡ tied optimum.


def _fill_probs_np(env: MarketMakingV1, action: int) -> tuple[float, float]:
    # (bid_tight, ask_tight) for each action id
    bid_tight = action in (ACTION_SYM, ACTION_FAVOR_BID)
    ask_tight = action in (ACTION_SYM, ACTION_FAVOR_ASK)
    p_bid = env.p_tight if bid_tight else env.p_wide
    p_ask = env.p_tight if ask_tight else env.p_wide
    return p_bid, p_ask


def _bid_ask_spreads(action: int) -> tuple[float, float]:
    # Must stay in sync with envs/market_making_v1.py._BID_SPREADS / _ASK_SPREADS.
    if action == ACTION_SYM:
        return 1.0, 1.0
    if action == ACTION_FAVOR_ASK:
        return 3.0, 1.0  # ask tight, bid wide → wide bid pays 3 when filled
    if action == ACTION_FAVOR_BID:
        return 1.0, 3.0
    raise ValueError(action)


def _build_transition_tables(env: MarketMakingV1) -> tuple[np.ndarray, np.ndarray]:
    """Return (P, R) with shape [n_states, n_actions, n_states] and [n_states, n_actions].

    P[s, a, s'] = transition probability from inventory s under action a.
    R[s, a]    = expected one-step reward.
    Bounded: fills that would cross ±I_max are rejected for that step
    (matches env.step logic exactly).
    """
    n_inv = env.n_inventory_states
    P = np.zeros((n_inv, N_ACTIONS, n_inv), dtype=np.float64)
    R = np.zeros((n_inv, N_ACTIONS), dtype=np.float64)
    kappa = env.inventory_penalty

    for s in range(n_inv):
        q = s - env.inventory_max
        for a in range(N_ACTIONS):
            p_bid, p_ask = _fill_probs_np(env, a)
            bid_spread, ask_spread = _bid_ask_spreads(a)

            bid_ok = q < env.inventory_max
            ask_ok = q > -env.inventory_max

            # Enumerate raw Bernoulli outcomes; rejected fills contribute no
            # inventory change and no capture (matches env.step).
            for bf_raw in (0, 1):
                pb = p_bid if bf_raw == 1 else (1.0 - p_bid)
                bf_eff = bf_raw if bid_ok else 0
                for af_raw in (0, 1):
                    pa = p_ask if af_raw == 1 else (1.0 - p_ask)
                    af_eff = af_raw if ask_ok else 0
                    prob = pb * pa
                    q_next = q + bf_eff - af_eff
                    capture = bf_eff * bid_spread + af_eff * ask_spread
                    inv_pen = kappa * (q_next ** 2)
                    reward = capture - inv_pen
                    s_next = q_next + env.inventory_max
                    P[s, a, s_next] += prob
                    R[s, a] += prob * reward

    return P, R


def _bellman_iterate(P: np.ndarray, R: np.ndarray, gamma: float, *, tol: float = 1e-10, max_iter: int = 5000) -> np.ndarray:
    n_inv, n_act, _ = P.shape
    V = np.zeros(n_inv, dtype=np.float64)
    for _ in range(max_iter):
        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def _expected_undiscounted_return_per_step(P: np.ndarray, R: np.ndarray, policy: np.ndarray) -> float:
    """Long-run average reward per step under `policy` (stationary distribution)."""
    n_inv = P.shape[0]
    P_pi = np.stack([P[s, policy[s], :] for s in range(n_inv)])
    # Stationary distribution of P_pi: solve π_d P_pi = π_d, sum = 1.
    A = P_pi.T - np.eye(n_inv)
    A = np.vstack([A, np.ones(n_inv)])
    b = np.zeros(n_inv + 1)
    b[-1] = 1.0
    pi_d, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi_d = np.clip(pi_d, 0.0, None)
    pi_d = pi_d / pi_d.sum()
    r_pi = np.array([R[s, policy[s]] for s in range(n_inv)])
    return float(np.dot(pi_d, r_pi))


def solve_analytical_as(env: MarketMakingV1) -> AnalyticalAS:
    P, R = _build_transition_tables(env)
    V = _bellman_iterate(P, R, env.gamma)
    Q = R + env.gamma * np.einsum("sat,t->sa", P, V)
    policy = np.argmax(Q, axis=1).astype(np.int64)

    action_probs = np.zeros((env.n_inventory_states, N_ACTIONS), dtype=np.float64)
    for s, a in enumerate(policy):
        action_probs[s, a] = 1.0
    skew = action_probs[:, ACTION_FAVOR_ASK] - action_probs[:, ACTION_FAVOR_BID]

    # Q-value margin per state: how much the optimum beats the 2nd-best.
    # Used downstream to mask out nearly-tied states from policy-shape checks,
    # because argmax-vs-softmax comparison is meaningless when the optimum is
    # a coin-flip (e.g. boundary states where one side can't fill).
    Q_sorted = np.sort(Q, axis=1)
    q_margin = Q_sorted[:, -1] - Q_sorted[:, -2]

    # Episode return from q=0 under optimal policy: we use the undiscounted
    # per-step expected reward times episode_length, matching the env (done
    # on t == episode_length, reward is undiscounted sum over steps). This
    # is what the training loop measures, so it is what the JSON's
    # `as_analytical_return` should be compared against.
    per_step = _expected_undiscounted_return_per_step(P, R, policy)
    expected_episode_return = per_step * env.episode_length

    return AnalyticalAS(
        V=V,
        policy=policy,
        action_probs=action_probs,
        skew=skew,
        expected_episode_return=expected_episode_return,
        q_margin=q_margin,
    )
