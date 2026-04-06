"""Analytical Market Making MDP with HMM Regime Switching.

Literature-conform model (Avellaneda-Stoikov / Guéant-Lehalle-Fernandez-Tapia
style) with discrete state/action spaces for exact Bellman equation solutions.

State: (regime, inventory)
  - regime ∈ {0=Noise, 1=Bull, 2=Bear}
  - inventory ∈ {-Q_max, ..., +Q_max}

Actions: discrete (bid_offset, ask_offset) tick pairs
  - Action 0: (2,2) tight symmetric — best spread capture (total offset 4)
  - Action 1: (1,9) aggressive long — strong directional (total offset 10)
  - Action 2: (9,1) aggressive short — strong directional (total offset 10)

a0 has a lower total offset → genuinely better spread capture. Directional
actions sacrifice spread for inventory flow. This creates a wide "a0 zone"
in Noise (where spread capture dominates) while Bull/Bear play pure
directional. The tradeoff is the Avellaneda-Stoikov insight: tighter quotes
capture more spread but expose you to adverse inventory in trending markets.

Fill model:
  p_fill(δ, regime) = arrival_rate(regime) * exp(-κ * (δ - 1))

Reward per step:
  r = p_bid * bid_edge + p_ask * ask_edge + inv * drift - φ * inv²

References:
  - Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
  - Guéant, Lehalle & Fernandez-Tapia (2013), "Dealing with the inventory risk"
  - Cartea, Jaimungal & Penalva (2015), "Algorithmic and HF Trading"
"""
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# HMM regime system
# ---------------------------------------------------------------------------

NOISE = 0
BULL = 1
BEAR = 2
N_REGIMES = 3

TRANSITION_MATRIX = np.array([
    [0.98, 0.01, 0.01],   # NOISE → ...
    [0.01, 0.98, 0.01],   # BULL  → ...
    [0.01, 0.01, 0.98],   # BEAR  → ...
])


# ---------------------------------------------------------------------------
# Action table (defaults — can be overridden via MDPConfig.actions)
# ---------------------------------------------------------------------------

ACTION_TABLE_ANALYTICAL = np.array([
    [2, 2],   # tight symmetric — best spread capture
    [1, 9],   # aggressive long — strong directional (bull)
    [9, 1],   # aggressive short — strong directional (bear)
])
N_ACTIONS_ANALYTICAL = len(ACTION_TABLE_ANALYTICAL)


# ---------------------------------------------------------------------------
# MDP specification
# ---------------------------------------------------------------------------

class MDPConfig(NamedTuple):
    """Parameters for the analytical market making MDP.

    Calibrated so that spread capture, directional drift, and inventory
    penalty are comparable in magnitude — producing policies that
    meaningfully differ across regimes.

    Three actions: tight symmetric (2,2) with total offset 4, and
    aggressive directional (1,9)/(9,1) with total offset 10. The lower
    total offset gives a0 genuinely better spread capture, creating a
    wide "a0 zone" in Noise. Moderate fill asymmetry (~3.3x ratio) lets
    the POMDP agent learn the regime over ~10 steps. Persistent regimes
    (98% self-transition, ~50 step duration) give time to exploit.
    Large drift (±1.0) makes playing a0 in trending regimes costly.
    The POMDP earns ~0.41/step more than regime-blind — an RL agent
    should aim to match or approach the POMDP trajectory.
    """
    max_inv: int = 10
    tick_size: float = 0.05
    half_spread_ticks: int = 2
    gamma: float = 0.90
    fill_decay: float = 0.30          # κ in exp(-κ * (δ - 1))
    # Per-regime arrival rates: (noise, bull, bear)
    # Market sells hit our bid; market buys hit our ask
    # Moderate asymmetry (~3.3x ratio) — regime learnable over ~10 steps
    sell_arrival: tuple = (0.30, 0.50, 0.15)
    buy_arrival: tuple = (0.30, 0.15, 0.50)
    # Per-regime mid-price drift (large — wrong-side inventory is very costly)
    drift: tuple = (0.0, 1.00, -1.00)
    # Quadratic inventory penalty coefficient
    inv_penalty: float = 0.001
    # HMM transition matrix (row = from, col = to); None = use default
    transition_matrix: tuple | None = None
    # Action table as flat tuple; None = use default ACTION_TABLE_ANALYTICAL
    actions: tuple | None = None
    # POMDP belief grid resolution
    n_belief_points: int = 21


class FillProbs(NamedTuple):
    """Fill probabilities per (regime, action)."""
    bid: np.ndarray   # (n_regimes, n_actions)
    ask: np.ndarray   # (n_regimes, n_actions)


class MDPTables(NamedTuple):
    """Pre-computed MDP tables for value iteration."""
    reward: np.ndarray       # (n_regimes, n_inv, n_actions)
    trans_inv: np.ndarray    # (n_regimes, n_inv, n_actions, n_inv)
    trans_regime: np.ndarray # (n_regimes, n_regimes)
    fill_probs: FillProbs
    config: MDPConfig


class VISolution(NamedTuple):
    """Value iteration output."""
    policy: np.ndarray    # int — optimal action per state
    values: np.ndarray    # float — state values
    Q: np.ndarray         # float — Q-values
    n_iters: int
    residuals: list       # Bellman residual per iteration


# ---------------------------------------------------------------------------
# Fill probability model
# ---------------------------------------------------------------------------

def _get_action_table(cfg: MDPConfig) -> np.ndarray:
    """Return the action table, either from config or the module default."""
    if cfg.actions is not None:
        n_act = len(cfg.actions) // 2
        return np.array(cfg.actions, dtype=float).reshape(n_act, 2)
    return ACTION_TABLE_ANALYTICAL.astype(float)


def compute_fill_probs(cfg: MDPConfig) -> FillProbs:
    """Avellaneda-Stoikov exponential fill model.

    p_fill(δ, regime) = arrival_rate(regime) * exp(-κ * (δ - 1))
    """
    kappa = cfg.fill_decay
    at = _get_action_table(cfg)

    sell_rates = np.array(cfg.sell_arrival)
    buy_rates = np.array(cfg.buy_arrival)

    bid_fill = sell_rates[:, None] * np.exp(-kappa * (at[None, :, 0] - 1))
    ask_fill = buy_rates[:, None] * np.exp(-kappa * (at[None, :, 1] - 1))

    return FillProbs(
        bid=np.clip(bid_fill, 0.0, 1.0),
        ask=np.clip(ask_fill, 0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Exact transition and reward tables
# ---------------------------------------------------------------------------

def build_mdp_tables(cfg: MDPConfig = MDPConfig()) -> MDPTables:
    """Build exact MDP transition and reward tables in closed form."""
    max_inv = cfg.max_inv
    n_inv = 2 * max_inv + 1
    tick = cfg.tick_size
    hs = cfg.half_spread_ticks
    at = _get_action_table(cfg)
    n_act = len(at)
    drift = np.array(cfg.drift)
    phi = cfg.inv_penalty
    T_hmm = (np.array(cfg.transition_matrix).reshape(N_REGIMES, N_REGIMES)
             if cfg.transition_matrix is not None
             else np.array(TRANSITION_MATRIX))

    fills = compute_fill_probs(cfg)
    inv_grid = np.arange(n_inv) - max_inv

    bid_edge = (at[:, 0] + hs) * tick  # (n_actions,)
    ask_edge = (at[:, 1] + hs) * tick

    # --- Reward table ---
    reward = np.zeros((N_REGIMES, n_inv, n_act))
    for r in range(N_REGIMES):
        for a in range(n_act):
            pb, pa = fills.bid[r, a], fills.ask[r, a]
            sc = pb * bid_edge[a] + pa * ask_edge[a]
            for qi in range(n_inv):
                q = inv_grid[qi]
                reward[r, qi, a] = sc + q * drift[r] - phi * q * q

    # --- Inventory transition tensor ---
    trans_inv = np.zeros((N_REGIMES, n_inv, n_act, n_inv))
    for r in range(N_REGIMES):
        for a in range(n_act):
            pb, pa = fills.bid[r, a], fills.ask[r, a]
            outcomes = [
                (0,  pb * pa + (1 - pb) * (1 - pa)),
                (+1, pb * (1 - pa)),
                (-1, (1 - pb) * pa),
            ]
            for qi in range(n_inv):
                q = inv_grid[qi]
                for dq, prob in outcomes:
                    q_new = int(np.clip(q + dq, -max_inv, max_inv))
                    trans_inv[r, qi, a, q_new + max_inv] += prob

    return MDPTables(
        reward=reward, trans_inv=trans_inv, trans_regime=T_hmm,
        fill_probs=fills, config=cfg,
    )


# ---------------------------------------------------------------------------
# Value iteration — full information (regime observed)
# ---------------------------------------------------------------------------

def solve_full_info(tables: MDPTables, verbose: bool = False) -> VISolution:
    """Solve the MDP when the agent observes the true regime.

    Bellman equation (regime transitions first):

        V(r, q) = max_a Σ_{r'} T(r,r')
                  [R(r',q,a) + γ Σ_{q'} P(q'|r',q,a) V(r',q')]

    63-state problem — converges in milliseconds.
    """
    R = tables.reward
    T_inv = tables.trans_inv
    T_hmm = tables.trans_regime
    gamma = tables.config.gamma

    V = np.zeros((N_REGIMES, R.shape[1]))
    residuals = []

    for it in range(5000):
        future = np.einsum('rqai,ri->rqa', T_inv, V)
        val_next = R + gamma * future
        Q = np.einsum('rs,sqa->rqa', T_hmm, val_next)

        V_new = np.max(Q, axis=-1)
        residual = float(np.max(np.abs(V_new - V)))
        residuals.append(residual)
        V = V_new

        if residual < 1e-10:
            if verbose:
                print(f"  Converged at iter {it+1} "
                      f"(residual={residual:.2e})")
            break

    return VISolution(policy=np.argmax(Q, axis=-1), values=V, Q=Q,
                      n_iters=it + 1, residuals=residuals)


# ---------------------------------------------------------------------------
# Value iteration — POMDP (regime hidden, belief-state MDP)
# ---------------------------------------------------------------------------

def _build_belief_grid(n_points: int) -> np.ndarray:
    """Uniform grid on the 3-regime belief simplex."""
    pts = []
    for i in range(n_points):
        for j in range(n_points - i):
            k = n_points - 1 - i - j
            pts.append(np.array([i, j, k], dtype=float) / (n_points - 1))
    return np.array(pts)


def _nearest_belief(b: np.ndarray, grid: np.ndarray) -> int:
    return int(np.argmin(np.sum(np.abs(grid - b), axis=1)))


def _bayesian_update(belief: np.ndarray, obs_bid: int, obs_ask: int,
                     action: int, fills: FillProbs,
                     T_hmm: np.ndarray) -> np.ndarray:
    """Bayesian belief update: predict via HMM, then condition on fills."""
    b_pred = T_hmm.T @ belief
    lik = np.ones(N_REGIMES)
    for r in range(N_REGIMES):
        pb, pa = fills.bid[r, action], fills.ask[r, action]
        lik[r] = (pb if obs_bid else 1 - pb) * (pa if obs_ask else 1 - pa)
    b_new = b_pred * lik
    s = b_new.sum()
    return b_new / s if s > 0 else np.ones(N_REGIMES) / N_REGIMES


def solve_pomdp_belief(tables: MDPTables, verbose: bool = False) -> VISolution:
    """Solve the POMDP via discretized belief-state value iteration.

    State = (belief, inventory).
    Observation = (bid_filled, ask_filled) — 4 outcomes.
    """
    R = tables.reward
    T_hmm = tables.trans_regime
    fills = tables.fill_probs
    gamma = tables.config.gamma
    max_inv = tables.config.max_inv
    n_inv = R.shape[1]
    n_act = R.shape[2]

    grid = _build_belief_grid(tables.config.n_belief_points)
    n_beliefs = len(grid)

    if verbose:
        print(f"  POMDP: {n_beliefs} belief × {n_inv} inv "
              f"= {n_beliefs * n_inv} states")

    obs_cases = [(0, 0, 0), (0, 1, -1), (1, 0, +1), (1, 1, 0)]

    # Pre-compute observation probabilities and belief transitions
    obs_prob = np.zeros((n_beliefs, n_act, 4))
    obs_next = np.zeros((n_beliefs, n_act, 4), dtype=int)

    for bi in range(n_beliefs):
        b = grid[bi]
        b_pred = T_hmm.T @ b
        for a in range(n_act):
            for oi, (ob, oa, _) in enumerate(obs_cases):
                p = sum(
                    b_pred[r] *
                    (fills.bid[r, a] if ob else 1 - fills.bid[r, a]) *
                    (fills.ask[r, a] if oa else 1 - fills.ask[r, a])
                    for r in range(N_REGIMES)
                )
                obs_prob[bi, a, oi] = p
                b_next = _bayesian_update(b, ob, oa, a, fills, T_hmm)
                obs_next[bi, a, oi] = _nearest_belief(b_next, grid)

    V = np.zeros((n_beliefs, n_inv))
    residuals = []
    dq_arr = np.array([c[2] for c in obs_cases])

    for it in range(2000):
        Q_all = np.zeros((n_beliefs, n_inv, n_act))
        for bi in range(n_beliefs):
            b_pred = T_hmm.T @ grid[bi]
            R_exp = np.einsum('r,rqa->qa', b_pred, R)

            for a in range(n_act):
                future = np.zeros(n_inv)
                for oi in range(4):
                    p = obs_prob[bi, a, oi]
                    if p < 1e-15:
                        continue
                    nbi = obs_next[bi, a, oi]
                    dq = dq_arr[oi]
                    for qi in range(n_inv):
                        qi_new = min(max(qi + dq, 0), n_inv - 1)
                        future[qi] += p * V[nbi, qi_new]
                Q_all[bi, :, a] = R_exp[:, a] + gamma * future

        V_new = np.max(Q_all, axis=-1)
        residual = float(np.max(np.abs(V_new - V)))
        residuals.append(residual)
        V = V_new

        if verbose and (it + 1) % 50 == 0:
            print(f"    iter {it+1}: residual={residual:.2e}")
        if residual < 1e-8:
            if verbose:
                print(f"  POMDP converged at iter {it+1} "
                      f"(residual={residual:.2e})")
            break

    return VISolution(policy=np.argmax(Q_all, axis=-1), values=V, Q=Q_all,
                      n_iters=it + 1, residuals=residuals)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def print_policy(sol: VISolution, tables: MDPTables):
    """Pretty-print the optimal policy table."""
    mi = tables.config.max_inv
    at = _get_action_table(tables.config)
    names = ["Noise", "Bull ", "Bear "]
    levels = list(range(-mi, mi + 1, max(1, mi // 5)))

    if sol.policy.ndim == 2 and sol.policy.shape[0] == N_REGIMES:
        header = "         " + "".join(f"{'q='+str(i):>8s}" for i in levels)
        print(header)
        for r in range(N_REGIMES):
            parts = []
            for q in levels:
                a = sol.policy[r, q + mi]
                parts.append(f"({int(at[a,0])},{int(at[a,1])})")
            print("  " + names[r] + "  " + "".join(f"{p:>8s}" for p in parts))


def print_fill_probs(tables: MDPTables):
    at = _get_action_table(tables.config)
    n_act = len(at)
    fills = tables.fill_probs
    names = ["Noise", "Bull", "Bear"]
    print("  Fill probabilities (p_bid / p_ask):")
    header = "         " + "".join(
        f"  a{a}=({int(at[a,0])},{int(at[a,1])}) "
        for a in range(n_act))
    print(header)
    for r in range(N_REGIMES):
        parts = [f"{fills.bid[r,a]:.3f}/{fills.ask[r,a]:.3f}"
                 for a in range(n_act)]
        print(f"  {names[r]:>5s}:  " + "    ".join(parts))


def stationary_distribution(tables: MDPTables) -> np.ndarray:
    """Stationary distribution of the HMM regime chain."""
    T = tables.trans_regime
    vals, vecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    return pi / pi.sum()


def value_of_info(tables: MDPTables, full_sol: VISolution,
                  pomdp_sol: VISolution) -> float:
    """Value of regime information at (stationary belief, zero inventory)."""
    q0 = tables.config.max_inv
    pi = stationary_distribution(tables)
    v_full = float(pi @ full_sol.values[:, q0])
    grid = _build_belief_grid(tables.config.n_belief_points)
    bi = _nearest_belief(pi, grid)
    return float(v_full - pomdp_sol.values[bi, q0])


# ---------------------------------------------------------------------------
# Episode simulation
# ---------------------------------------------------------------------------

class SimResult(NamedTuple):
    """Output of simulate_episodes."""
    cumulative_reward: np.ndarray  # (n_episodes, n_steps)
    mean_reward: np.ndarray        # (n_steps,) — mean cumulative across episodes
    std_reward: np.ndarray         # (n_steps,) — std of cumulative


def simulate_episodes(
    tables: MDPTables,
    policy: str,
    full_sol: VISolution | None = None,
    pomdp_sol: VISolution | None = None,
    n_episodes: int = 500,
    n_steps: int = 200,
    seed: int = 42,
    locked_regime: int = -1,
) -> SimResult:
    """Simulate episodes under a given policy type.

    Args:
        policy: one of "full_info", "pomdp", "blind"
        full_sol: required if policy == "full_info"
        pomdp_sol: required if policy == "pomdp"
        locked_regime: -1 for normal HMM, 0/1/2 to lock regime
    """
    rng = np.random.default_rng(seed)
    cfg = tables.config
    max_inv = cfg.max_inv
    at = _get_action_table(cfg)
    tick = cfg.tick_size
    hs = cfg.half_spread_ticks
    drift = np.array(cfg.drift)
    phi = cfg.inv_penalty
    T_hmm = tables.trans_regime
    fills = tables.fill_probs

    # POMDP belief grid (only needed for pomdp policy)
    grid = None
    if policy == "pomdp":
        assert pomdp_sol is not None
        grid = _build_belief_grid(cfg.n_belief_points)

    # Stationary distribution for initial regime sampling
    pi = stationary_distribution(tables)

    cumulative = np.zeros((n_episodes, n_steps))

    for ep in range(n_episodes):
        # Sample initial regime
        if locked_regime >= 0:
            regime = locked_regime
        else:
            regime = rng.choice(N_REGIMES, p=pi)
        inv = 0
        belief = pi.copy() if policy == "pomdp" else None
        cum_r = 0.0

        for t in range(n_steps):
            # Select action
            qi = inv + max_inv
            if policy == "full_info":
                action = int(full_sol.policy[regime, qi])
            elif policy == "pomdp":
                bi = _nearest_belief(belief, grid)
                action = int(pomdp_sol.policy[bi, qi])
            else:  # blind
                action = 0

            # Sample fills
            pb = fills.bid[regime, action]
            pa = fills.ask[regime, action]
            bid_filled = rng.random() < pb
            ask_filled = rng.random() < pa

            # Compute reward
            bid_edge = (at[action, 0] + hs) * tick
            ask_edge = (at[action, 1] + hs) * tick
            reward = (bid_filled * bid_edge + ask_filled * ask_edge
                      + inv * drift[regime] - phi * inv * inv)

            cum_r += reward
            cumulative[ep, t] = cum_r

            # Update inventory
            dq = int(bid_filled) - int(ask_filled)
            inv = max(-max_inv, min(max_inv, inv + dq))

            # Update belief (POMDP only)
            if policy == "pomdp":
                belief = _bayesian_update(
                    belief, int(bid_filled), int(ask_filled),
                    action, fills, T_hmm)

            # Regime transition (skip if locked)
            if locked_regime < 0:
                regime = rng.choice(N_REGIMES, p=T_hmm[regime])

    return SimResult(
        cumulative_reward=cumulative,
        mean_reward=cumulative.mean(axis=0),
        std_reward=cumulative.std(axis=0),
    )
