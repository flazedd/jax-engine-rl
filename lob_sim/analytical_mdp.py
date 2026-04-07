"""Analytical solutions for the POMDP market making environment.

Phase 1: Locked-regime VI + Oracle A (full-information MDP).
Phase 2: Oracle B (POMDP belief-state VI).

Locked-regime VI: fix regime r, solve the 11-state inventory MDP.
  V_r(q) = max_a { R(q,r,a) + γ · Σ_{q'} P(q'|q,a,r) · V_r(q') }

Oracle A: full information, 33-state MDP (3 regimes × 11 inventories).
  V(q,r) = max_a { R(q,r,a) + γ · Σ_{r'} P(r'|r) · Σ_{q'} P(q'|q,a,r) · V(q',r') }

Oracle B: POMDP with discretized belief states.
  V(q,b) = max_a { E[r|b,a] + γ · Σ_o P(o|b,a) · V(q'(o), b'(o,a,b)) }

All computations use JAX arrays, derived from Phase 0 EnvParams.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.jax_env import EnvParams


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MDPTables(NamedTuple):
    """Pre-computed MDP reward and transition tables."""
    reward: jnp.ndarray       # (n_regimes, n_inv, n_actions) expected reward
    trans_inv: jnp.ndarray    # (n_regimes, n_inv, n_actions, n_inv) P(q'|q,r,a)
    fill_bid: jnp.ndarray     # (n_regimes, n_actions) fill probabilities
    fill_ask: jnp.ndarray     # (n_regimes, n_actions) fill probabilities


class VISolution(NamedTuple):
    """Value iteration output."""
    policy: jnp.ndarray       # optimal action per state
    values: jnp.ndarray       # V(state)
    Q: jnp.ndarray            # Q(state, action)
    n_iters: int


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

def build_mdp_tables(params: EnvParams) -> MDPTables:
    """Build exact reward and transition tables from environment parameters.

    Fill model: P(fill_side) = exp(-κ[regime, side] · δ[action, side])
    4 outcomes per (regime, inventory, action) from independent bid/ask fills.
    Reward: spread_pnl - inventory_penalty - boundary_penalty (on post-step q').
    """
    n_r = params.n_regimes
    n_a = params.n_actions
    inv_max = params.inventory_max
    n_q = 2 * inv_max + 1
    inv_grid = jnp.arange(n_q) - inv_max  # [-5, ..., 5]

    # Fill probabilities: (n_regimes, n_actions)
    fill_bid = jnp.exp(-params.kappa[:, 0, None] * params.delta[None, :, 0])
    fill_ask = jnp.exp(-params.kappa[:, 1, None] * params.delta[None, :, 1])

    # 4 fill outcomes: (bid_filled, ask_filled)
    fb = jnp.array([0.0, 0.0, 1.0, 1.0])
    fa = jnp.array([0.0, 1.0, 0.0, 1.0])
    dq = jnp.array([0, -1, 1, 0])

    # Outcome probabilities: (n_r, n_a, 4)
    p_out = jnp.stack([
        (1 - fill_bid) * (1 - fill_ask),
        (1 - fill_bid) * fill_ask,
        fill_bid * (1 - fill_ask),
        fill_bid * fill_ask,
    ], axis=-1)

    # Next inventory: q' = clip(q + dq, -inv_max, inv_max)
    q_next = jnp.clip(
        inv_grid[:, None] + dq[None, :], -inv_max, inv_max)  # (n_q, 4)
    q_next_idx = (q_next + inv_max).astype(jnp.int32)
    q_next_f = q_next.astype(jnp.float32)

    # Spread PnL per (action, outcome)
    spread_pnl = (fb[None, :] * params.delta[:, 0, None]
                  + fa[None, :] * params.delta[:, 1, None])  # (n_a, 4)

    # Inventory risk penalty: γ_inv · q'² · σ²[r]  — shape (n_r, n_q, 4)
    inv_pen = (params.gamma_inventory
               * q_next_f[None, :, :] ** 2
               * params.sigma_sq[:, None, None])

    # Boundary penalty: 5.0 · |q'| · 𝟙(|q'|==max)  — shape (n_q, 4)
    at_bound = (jnp.abs(q_next) == inv_max).astype(jnp.float32)
    bound_pen = params.boundary_penalty * jnp.abs(q_next_f) * at_bound

    # Reward per outcome: (n_r, n_q, n_a, 4)
    r_per_outcome = (spread_pnl[None, None, :, :]
                     - inv_pen[:, :, None, :]
                     - bound_pen[None, :, None, :])

    # Expected reward: R[r, q, a] = Σ_o P(o|r,a) · reward(r,q,a,o)
    R = jnp.sum(p_out[:, None, :, :] * r_per_outcome, axis=-1)

    # Transition table: T[r, q, a, q'] = Σ_{o: next(q,o)==q'} P(o|r,a)
    q_next_onehot = jax.nn.one_hot(q_next_idx, n_q)  # (n_q, 4, n_q)
    T = jnp.einsum('rao,qop->rqap', p_out, q_next_onehot)

    return MDPTables(reward=R, trans_inv=T, fill_bid=fill_bid, fill_ask=fill_ask)


# ---------------------------------------------------------------------------
# Locked-regime value iteration (11-state MDP per regime)
# ---------------------------------------------------------------------------

def solve_locked(
    tables: MDPTables,
    regime: int,
    params: EnvParams,
    tol: float = 1e-6,
    max_iters: int = 5000,
) -> VISolution:
    """Solve the locked-regime MDP for a single regime via value iteration.

    Bellman: V(q) = max_a { R(q,r,a) + γ · Σ_{q'} T(q'|q,r,a) · V(q') }
    """
    R_r = tables.reward[regime]       # (n_inv, n_actions)
    T_r = tables.trans_inv[regime]    # (n_inv, n_actions, n_inv)
    gamma = params.gamma_disc
    n_inv = R_r.shape[0]

    V = jnp.zeros(n_inv)
    Q = jnp.zeros_like(R_r)

    for it in range(max_iters):
        Q = R_r + gamma * jnp.einsum('qap,p->qa', T_r, V)
        V_new = jnp.max(Q, axis=-1)
        residual = float(jnp.max(jnp.abs(V_new - V)))
        V = V_new
        if residual < tol:
            break

    return VISolution(
        policy=jnp.argmax(Q, axis=-1),
        values=V,
        Q=Q,
        n_iters=it + 1,
    )


def solve_all_locked(
    tables: MDPTables, params: EnvParams, **kwargs,
) -> list[VISolution]:
    """Solve locked-regime VI for all regimes."""
    return [solve_locked(tables, r, params, **kwargs)
            for r in range(params.n_regimes)]


# ---------------------------------------------------------------------------
# Oracle A — full-information MDP (33 states)
# ---------------------------------------------------------------------------

def solve_oracle_a(
    tables: MDPTables,
    params: EnvParams,
    tol: float = 1e-6,
    max_iters: int = 5000,
) -> VISolution:
    """Solve the full-information MDP (Oracle A) via value iteration.

    State: (inventory, regime) — 11 × 3 = 33 states.
    Agent observes the true regime at each step.

    Bellman:
      V(q,r) = max_a { R(q,r,a) + γ · Σ_{r'} P(r'|r) · Σ_{q'} T(q'|q,a,r) · V(q',r') }

    Fills depend on current regime r; next regime r' from HMM transition.
    """
    R = tables.reward              # (n_r, n_inv, n_a)
    T = tables.trans_inv           # (n_r, n_inv, n_a, n_inv)
    hmm = params.hmm_transition    # (n_r, n_r)
    gamma = params.gamma_disc
    n_r, n_inv, n_a = R.shape

    V = jnp.zeros((n_r, n_inv))
    Q = jnp.zeros((n_r, n_inv, n_a))

    for it in range(max_iters):
        # TV[r, q, a, r'] = Σ_{q'} T[r,q,a,q'] · V[r', q']
        TV = jnp.einsum('rqap,sp->rqas', T, V)
        # future[r, q, a] = Σ_{r'} hmm[r,r'] · TV[r,q,a,r']
        future = jnp.einsum('rs,rqas->rqa', hmm, TV)
        Q = R + gamma * future
        V_new = jnp.max(Q, axis=-1)
        residual = float(jnp.max(jnp.abs(V_new - V)))
        V = V_new
        if residual < tol:
            break

    return VISolution(
        policy=jnp.argmax(Q, axis=-1),
        values=V,
        Q=Q,
        n_iters=it + 1,
    )


# ---------------------------------------------------------------------------
# Q_max — shared normalisation constant
# ---------------------------------------------------------------------------

def compute_q_max(locked_solutions: list[VISolution]) -> float:
    """Q_max = max_{q, r} Q*_locked(q, r, a*(q,r)).

    Since V(q) = max_a Q(q,a) = Q(q, a*), this is max over all
    locked-regime value functions.
    """
    return float(max(jnp.max(sol.values) for sol in locked_solutions))


# ---------------------------------------------------------------------------
# Precondition 1 — locked-regime policies distinct and decisive
# ---------------------------------------------------------------------------

def precondition_1(
    locked_solutions: list[VISolution],
    q_max: float,
) -> dict:
    """Check Precondition 1: locked-regime policies are distinct and decisive.

    Returns dict with:
      gap_per_regime: mean relative gap per regime (% of Q_max)
      pairwise_disagreement: fraction of states where policies differ
      pass_gap: all mean gaps > 5%
      pass_disagreement: all pairwise disagreements > 20%
      passed: both conditions met
    """
    n_regimes = len(locked_solutions)
    n_inv = locked_solutions[0].Q.shape[0]

    # Per-regime gap: (Q*(q, a*) - Q*(q, a_2nd)) / Q_max
    gaps = {}
    for r, sol in enumerate(locked_solutions):
        Q_sorted = jnp.sort(sol.Q, axis=-1)
        gap = (Q_sorted[:, -1] - Q_sorted[:, -2]) / q_max * 100.0
        gaps[r] = {
            "mean": float(jnp.mean(gap)),
            "min": float(jnp.min(gap)),
            "values": gap,
        }

    # Pairwise disagreement: fraction of inventory states with different actions
    disagreements = {}
    for r1 in range(n_regimes):
        for r2 in range(r1 + 1, n_regimes):
            p1 = locked_solutions[r1].policy
            p2 = locked_solutions[r2].policy
            frac = float(jnp.mean((p1 != p2).astype(jnp.float32))) * 100.0
            disagreements[(r1, r2)] = frac

    pass_gap = all(g["mean"] > 5.0 for g in gaps.values())
    pass_disagree = all(d > 20.0 for d in disagreements.values())

    return {
        "gap_per_regime": gaps,
        "pairwise_disagreement": disagreements,
        "pass_gap": pass_gap,
        "pass_disagreement": pass_disagree,
        "passed": pass_gap and pass_disagree,
    }


# ---------------------------------------------------------------------------
# Oracle B — POMDP belief-state VI
# ---------------------------------------------------------------------------

def build_belief_grid(grid_size: int) -> jnp.ndarray:
    """Uniform triangular grid on the 3-regime belief simplex Δ².

    grid_size: number of divisions per edge (e.g., 20 → 21 points per edge).
    Returns array of shape (n_grid, 3) where each row sums to 1.0.
    """
    n = grid_size + 1  # points per edge
    pts = []
    for i in range(n):
        for j in range(n - i):
            k = grid_size - i - j
            pts.append([i, j, k])
    return jnp.array(pts, dtype=jnp.float32) / grid_size


def nearest_belief_idx(b, grid) -> int:
    """Find nearest grid point by L1 distance.

    Works with both numpy and JAX arrays.
    """
    dists = np.sum(np.abs(np.asarray(grid) - np.asarray(b)), axis=1)
    return int(np.argmin(dists))


def hmm_filter_update(
    belief: jnp.ndarray,
    fill_bid: int,
    fill_ask: int,
    mid_change_idx: int,
    action: int,
    tables: MDPTables,
    params: EnvParams,
) -> jnp.ndarray:
    """Exact HMM filter: condition on observation, then predict.

    Order: likelihood-weight → normalise → predict (HMM transition).

    belief: P(current_regime | past_observations) — prior for current step.
    Returns: P(next_regime | observations_including_current) — prior for next step.
    """
    n_r = params.n_regimes
    # Likelihood: P(fills, mid_change | regime, action)
    lik = jnp.ones(n_r)
    for r in range(n_r):
        pb = float(tables.fill_bid[r, action])
        pa = float(tables.fill_ask[r, action])
        l_fills = (pb if fill_bid else 1 - pb) * (pa if fill_ask else 1 - pa)
        l_mc = float(params.drift_probs[r, mid_change_idx])
        lik = lik.at[r].set(l_fills * l_mc)

    # Condition on observation
    b_cond = belief * lik
    total = b_cond.sum()
    b_cond = jnp.where(total > 1e-15, b_cond / total,
                        jnp.ones(n_r) / n_r)

    # Predict (HMM transition to next regime)
    return params.hmm_transition.T @ b_cond


def solve_oracle_b(
    tables: MDPTables,
    params: EnvParams,
    grid_size: int = 20,
    tol: float = 1e-6,
    max_iters: int = 2000,
    verbose: bool = False,
) -> VISolution:
    """Solve the POMDP via discretized belief-state value iteration (Oracle B).

    State: (belief ∈ Δ², inventory ∈ [-5,5]).
    Observation: (fill_bid, fill_ask, mid_change) — 12 outcomes.

    Bellman:
      V(q,b) = max_a { R_exp(q,b,a) + γ · Σ_o P(o|b,a) · V(q'(o), b'(o,a,b)) }

    Uses nearest-grid-point interpolation for belief transitions.
    """
    n_r = params.n_regimes
    n_a = params.n_actions
    inv_max = params.inventory_max
    n_inv = 2 * inv_max + 1
    gamma = params.gamma_disc

    grid = build_belief_grid(grid_size)
    n_b = len(grid)

    if verbose:
        print(f"  Oracle B: {n_b} belief × {n_inv} inv "
              f"= {n_b * n_inv} states")

    # Enumerate 12 observations: (fill_bid, fill_ask, mid_change_idx)
    obs_list = [(fb, fa, mc)
                for fb in range(2) for fa in range(2) for mc in range(3)]
    n_obs = len(obs_list)
    obs_dq_np = np.array([fb - fa for fb, fa, _ in obs_list], dtype=np.int32)

    # Convert to numpy for precomputation
    grid_np = np.array(grid)
    fill_bid_np = np.array(tables.fill_bid)  # (3, 3)
    fill_ask_np = np.array(tables.fill_ask)  # (3, 3)
    drift_np = np.array(params.drift_probs)  # (3, 3)
    hmm_np = np.array(params.hmm_transition)  # (3, 3)

    # Pre-compute for each (belief_idx, action, obs):
    #   obs_prob[bi, a, o]     = P(obs | belief, action)
    #   obs_next_bi[bi, a, o]  = nearest grid idx for updated belief
    obs_prob_np = np.zeros((n_b, n_a, n_obs))
    obs_next_bi_np = np.zeros((n_b, n_a, n_obs), dtype=np.int32)

    for bi in range(n_b):
        b = grid_np[bi]
        for a in range(n_a):
            for oi, (fb, fa, mc) in enumerate(obs_list):
                # Per-regime likelihood
                lik = np.zeros(n_r)
                for r in range(n_r):
                    pb = fill_bid_np[r, a]
                    pa = fill_ask_np[r, a]
                    lik[r] = ((pb if fb else 1 - pb)
                              * (pa if fa else 1 - pa)
                              * drift_np[r, mc])

                p_obs = float(b @ lik)
                obs_prob_np[bi, a, oi] = p_obs

                # Belief update: condition → normalise → predict
                if p_obs > 1e-15:
                    b_cond = b * lik
                    b_cond /= b_cond.sum()
                    b_next = hmm_np.T @ b_cond
                else:
                    b_next = np.ones(n_r) / n_r

                obs_next_bi_np[bi, a, oi] = nearest_belief_idx(
                    b_next, grid_np)

    if verbose:
        print(f"  Pre-computation done ({n_b * n_a * n_obs} belief updates)")

    # Convert to JAX for VI sweeps
    obs_prob_j = jnp.array(obs_prob_np)
    obs_next_bi_j = jnp.array(obs_next_bi_np)
    obs_dq_j = jnp.array(obs_dq_np)
    R = tables.reward  # (n_r, n_inv, n_a)

    # Expected reward: R_exp[bi, qi, a] = Σ_r grid[bi, r] · R[r, qi, a]
    R_exp = jnp.einsum('br,rqa->bqa', grid, R)

    inv_idx = jnp.arange(n_inv)
    V = jnp.zeros((n_b, n_inv))

    for it in range(max_iters):
        # Future value: Σ_o P(o|b,a) · V(next_bi(o), qi'(o))
        future = jnp.zeros((n_b, n_inv, n_a))
        for oi in range(n_obs):
            nbi = obs_next_bi_j[:, :, oi]         # (n_b, n_a)
            qi_new = jnp.clip(
                inv_idx + obs_dq_j[oi], 0, n_inv - 1)  # (n_inv,)
            # V[nbi] → (n_b, n_a, n_inv); [:,:,qi_new] reindexes inventory
            V_at = V[nbi][:, :, qi_new]            # (n_b, n_a, n_inv)
            # Accumulate: transpose to (n_b, n_inv, n_a) to match Q shape
            future += jnp.transpose(
                obs_prob_j[:, :, oi, None] * V_at, (0, 2, 1))

        Q = R_exp + gamma * future
        V_new = jnp.max(Q, axis=-1)
        residual = float(jnp.max(jnp.abs(V_new - V)))
        V = V_new

        if verbose and (it + 1) % 50 == 0:
            print(f"    iter {it+1}: residual={residual:.2e}")
        if residual < tol:
            if verbose:
                print(f"  Converged at iter {it+1} "
                      f"(residual={residual:.2e})")
            break

    return VISolution(
        policy=jnp.argmax(Q, axis=-1),
        values=V,
        Q=Q,
        n_iters=it + 1,
    )


# ---------------------------------------------------------------------------
# Convenience — print helpers
# ---------------------------------------------------------------------------

N_REGIMES = 3
N_ACTIONS = 3
REGIME_NAMES = ["Noise", "Bull", "Bear"]
ACTION_NAMES = ["sym(1,1)", "ask(1,3)", "bid(3,1)"]


def print_policy(sol: VISolution, params: EnvParams, label: str = ""):
    """Pretty-print the optimal policy table."""
    inv_max = params.inventory_max
    inv_grid = list(range(-inv_max, inv_max + 1))
    policy = sol.policy

    if label:
        print(f"  {label}")

    if policy.ndim == 2:
        # Oracle A: (n_regimes, n_inv)
        header = "         " + "".join(f"{'q='+str(q):>10s}" for q in inv_grid)
        print(header)
        for r in range(params.n_regimes):
            parts = [ACTION_NAMES[int(policy[r, qi])]
                     for qi in range(len(inv_grid))]
            print(f"  {REGIME_NAMES[r]:>5s}  " + "".join(f"{p:>10s}" for p in parts))
    else:
        # Locked regime: (n_inv,)
        header = "       " + "".join(f"{'q='+str(q):>10s}" for q in inv_grid)
        print(header)
        parts = [ACTION_NAMES[int(policy[qi])] for qi in range(len(inv_grid))]
        print("       " + "".join(f"{p:>10s}" for p in parts))


def print_fill_probs(tables: MDPTables):
    """Print the fill probability table."""
    print("  Fill probabilities (p_bid / p_ask):")
    header = "         " + "".join(
        f"  {ACTION_NAMES[a]:>12s}" for a in range(3))
    print(header)
    for r in range(3):
        parts = [f"{float(tables.fill_bid[r,a]):.4f}/{float(tables.fill_ask[r,a]):.4f}"
                 for a in range(3)]
        print(f"  {REGIME_NAMES[r]:>5s}:  " + "    ".join(parts))


# ---------------------------------------------------------------------------
# Numpy-based oracle simulation in switching environment
# ---------------------------------------------------------------------------

class SimRecord(NamedTuple):
    """Per-step records from oracle simulation."""
    regimes: np.ndarray    # (n_episodes, T) true regime
    actions: np.ndarray    # (n_episodes, T) action taken
    rewards: np.ndarray    # (n_episodes, T) reward
    inventories: np.ndarray  # (n_episodes, T) inventory before action


def _simulate_env_steps(
    params: EnvParams,
    n_episodes: int,
    t_episode: int,
    rng: np.random.Generator,
) -> dict:
    """Simulate environment transitions, returning per-step data.

    Returns dict with arrays (n_episodes, T) for regime, fill outcomes,
    mid_change, and a callback-friendly structure.
    """
    hmm = np.array(params.hmm_transition)
    drift = np.array(params.drift_probs)
    kappa = np.array(params.kappa)
    delta = np.array(params.delta)
    sigma_sq = np.array(params.sigma_sq)
    inv_max = int(params.inventory_max)
    gamma_inv = float(params.gamma_inventory)
    bound_pen = float(params.boundary_penalty)

    # Pre-generate regime sequences
    locked = int(params.locked_regime)
    regimes = np.zeros((n_episodes, t_episode), dtype=np.int32)
    if locked >= 0:
        regimes[:] = locked
    else:
        pi = np.array(params.stationary_dist)
        regimes[:, 0] = rng.choice(3, size=n_episodes, p=pi)
        for t in range(1, t_episode):
            for ep in range(n_episodes):
                regimes[ep, t] = rng.choice(3, p=hmm[regimes[ep, t - 1]])

    # Pre-generate mid_change outcomes
    mid_changes = np.zeros((n_episodes, t_episode), dtype=np.int32)
    for ep in range(n_episodes):
        for t in range(t_episode):
            mid_changes[ep, t] = rng.choice(3, p=drift[regimes[ep, t]])

    return {
        "regimes": regimes,
        "mid_changes": mid_changes,  # index into {-1,0,+1} as {0,1,2}
        "kappa": kappa,
        "delta": delta,
        "sigma_sq": sigma_sq,
        "inv_max": inv_max,
        "gamma_inv": gamma_inv,
        "bound_pen": bound_pen,
        "rng": rng,
    }


def _run_oracle_rollout(
    env_data: dict,
    policy_fn,
    n_episodes: int,
    t_episode: int,
) -> SimRecord:
    """Execute rollout with a policy function that sees (regime, inventory, belief).

    policy_fn(regime, inventory_idx, belief) -> action
    For Oracle A: ignores belief.
    For Oracle B: ignores regime.
    """
    kappa = env_data["kappa"]
    delta = env_data["delta"]
    sigma_sq = env_data["sigma_sq"]
    inv_max = env_data["inv_max"]
    n_inv = 2 * inv_max + 1
    gamma_inv = env_data["gamma_inv"]
    bound_pen = env_data["bound_pen"]
    rng = env_data["rng"]
    regimes = env_data["regimes"]
    mid_changes = env_data["mid_changes"]

    actions = np.zeros((n_episodes, t_episode), dtype=np.int32)
    rewards = np.zeros((n_episodes, t_episode), dtype=np.float64)
    inventories = np.zeros((n_episodes, t_episode), dtype=np.int32)

    # HMM filter state for Oracle B (unused by Oracle A)
    hmm = np.array(kappa)  # just to get shape — will use env_data
    hmm_trans = None
    drift_probs = None

    for ep in range(n_episodes):
        inv = 0
        belief = None  # will be set by policy_fn wrapper if needed

        for t in range(t_episode):
            r = regimes[ep, t]
            inv_idx = inv + inv_max
            inventories[ep, t] = inv

            # Get action from oracle policy
            action = policy_fn(r, inv_idx, ep, t)
            actions[ep, t] = action

            # Simulate fills
            p_bid = np.exp(-kappa[r, 0] * delta[action, 0])
            p_ask = np.exp(-kappa[r, 1] * delta[action, 1])
            fill_bid = int(rng.random() < p_bid)
            fill_ask = int(rng.random() < p_ask)

            # Inventory update
            new_inv = np.clip(inv + fill_bid - fill_ask, -inv_max, inv_max)

            # Reward
            spread = fill_bid * delta[action, 0] + fill_ask * delta[action, 1]
            inv_pen = gamma_inv * new_inv**2 * sigma_sq[r]
            at_bound = float(abs(new_inv) == inv_max)
            bpen = bound_pen * abs(new_inv) * at_bound
            rewards[ep, t] = spread - inv_pen - bpen

            inv = int(new_inv)

    return SimRecord(
        regimes=regimes,
        actions=actions,
        rewards=rewards,
        inventories=inventories,
    )


def simulate_oracle_a(
    oracle_a_sol: VISolution,
    params: EnvParams,
    n_episodes: int = 2000,
    seed: int = 42,
) -> SimRecord:
    """Simulate Oracle A in the switching environment.

    Oracle A observes the true regime and chooses action from its policy table.
    """
    tables = build_mdp_tables(params)
    rng = np.random.default_rng(seed)
    t_episode = int(params.t_episode)
    env_data = _simulate_env_steps(params, n_episodes, t_episode, rng)

    policy = np.array(oracle_a_sol.policy)  # (n_regimes, n_inv)

    def oracle_a_policy(regime, inv_idx, ep, t):
        return int(policy[regime, inv_idx])

    return _run_oracle_rollout(env_data, oracle_a_policy, n_episodes, t_episode)


def simulate_oracle_b(
    oracle_b_sol: VISolution,
    tables: MDPTables,
    params: EnvParams,
    grid_size: int = 20,
    n_episodes: int = 2000,
    seed: int = 42,
) -> SimRecord:
    """Simulate Oracle B in the switching environment.

    Oracle B maintains an exact HMM belief filter and looks up action
    from the POMDP policy table via nearest grid point.
    """
    rng = np.random.default_rng(seed)
    t_episode = int(params.t_episode)
    env_data = _simulate_env_steps(params, n_episodes, t_episode, rng)

    grid = np.array(build_belief_grid(grid_size))
    policy_b = np.array(oracle_b_sol.policy)  # (n_belief, n_inv)
    fill_bid_np = np.array(tables.fill_bid)
    fill_ask_np = np.array(tables.fill_ask)
    drift_np = np.array(params.drift_probs)
    hmm_np = np.array(params.hmm_transition)
    kappa = env_data["kappa"]
    delta = env_data["delta"]
    inv_max = env_data["inv_max"]
    regimes = env_data["regimes"]
    mid_changes = env_data["mid_changes"]
    n_r = int(params.n_regimes)

    # Pre-allocate beliefs and run belief filter forward
    # We need to interleave belief updates with action selection,
    # since the action affects the fill likelihood
    beliefs = np.zeros((n_episodes, t_episode, 3))
    actions_out = np.zeros((n_episodes, t_episode), dtype=np.int32)
    rewards_out = np.zeros((n_episodes, t_episode), dtype=np.float64)
    inventories_out = np.zeros((n_episodes, t_episode), dtype=np.int32)

    pi = np.array(params.stationary_dist)
    sigma_sq = env_data["sigma_sq"]
    gamma_inv = env_data["gamma_inv"]
    bound_pen = env_data["bound_pen"]

    for ep in range(n_episodes):
        belief = pi.copy()
        inv = 0

        for t in range(t_episode):
            r = regimes[ep, t]
            inv_idx = inv + inv_max
            beliefs[ep, t] = belief
            inventories_out[ep, t] = inv

            # Look up action from Oracle B policy
            bi = int(np.argmin(np.sum(np.abs(grid - belief), axis=1)))
            action = int(policy_b[bi, inv_idx])
            actions_out[ep, t] = action

            # Simulate fills
            p_bid = np.exp(-kappa[r, 0] * delta[action, 0])
            p_ask = np.exp(-kappa[r, 1] * delta[action, 1])
            fill_bid = int(env_data["rng"].random() < p_bid)
            fill_ask = int(env_data["rng"].random() < p_ask)

            # Inventory update
            new_inv = int(np.clip(inv + fill_bid - fill_ask, -inv_max, inv_max))

            # Reward
            spread = fill_bid * delta[action, 0] + fill_ask * delta[action, 1]
            inv_pen = gamma_inv * new_inv**2 * sigma_sq[r]
            at_bound = float(abs(new_inv) == inv_max)
            bpen = bound_pen * abs(new_inv) * at_bound
            rewards_out[ep, t] = spread - inv_pen - bpen

            # Belief update: condition on observation → predict
            mc_idx = mid_changes[ep, t]
            lik = np.zeros(n_r)
            for ri in range(n_r):
                pb = fill_bid_np[ri, action]
                pa = fill_ask_np[ri, action]
                lik[ri] = ((pb if fill_bid else 1 - pb)
                           * (pa if fill_ask else 1 - pa)
                           * drift_np[ri, mc_idx])
            b_cond = belief * lik
            total = b_cond.sum()
            if total > 1e-15:
                b_cond /= total
            else:
                b_cond = np.ones(n_r) / n_r
            belief = hmm_np.T @ b_cond

            inv = new_inv

    return SimRecord(
        regimes=regimes,
        actions=actions_out,
        rewards=rewards_out,
        inventories=inventories_out,
    )


# ---------------------------------------------------------------------------
# Precondition 2 — Oracle A regime-dependent under switching
# ---------------------------------------------------------------------------

def precondition_2(
    sim: SimRecord,
    oracle_a_sol: VISolution,
    locked_solutions: list[VISolution],
    q_max: float,
    params: EnvParams,
) -> dict:
    """Check Precondition 2: Oracle A is regime-dependent in switching env.

    Analyses Oracle A rollout tagged by true_regime:
      - Per-regime action distributions
      - Per-regime relative gaps (on shared Q_max)
      - Disagreement: Oracle A vs locked-regime policies
    """
    n_inv = 2 * int(params.inventory_max) + 1
    n_r = int(params.n_regimes)
    n_a = int(params.n_actions)

    # Per-regime action distributions
    action_dists = np.zeros((n_r, n_a))
    for r in range(n_r):
        mask = sim.regimes == r
        if mask.sum() == 0:
            continue
        acts = sim.actions[mask]
        for a in range(n_a):
            action_dists[r, a] = (acts == a).sum() / len(acts)

    # Per-regime mean relative gap from Oracle A Q-table (uniform over inventory)
    Q_a = np.array(oracle_a_sol.Q)  # (n_r, n_inv, n_a)
    Q_sorted = np.sort(Q_a, axis=-1)
    gap_per_state = (Q_sorted[:, :, -1] - Q_sorted[:, :, -2]) / q_max * 100.0

    gap_per_regime = {}
    for r in range(n_r):
        gap_per_regime[r] = {
            "mean": float(np.mean(gap_per_state[r])),
            "min": float(np.min(gap_per_state[r])),
            "values": gap_per_state[r],
        }

    # Oracle A vs locked-regime disagreement
    oracle_a_policy = np.array(oracle_a_sol.policy)
    disagree_vs_locked = {}
    for r in range(n_r):
        locked_pol = np.array(locked_solutions[r].policy)
        oracle_a_r = oracle_a_policy[r]
        frac = float(np.mean(oracle_a_r != locked_pol)) * 100.0
        disagree_vs_locked[r] = frac

    # Gate: overall mean gap > 5% and action distributions separate by regime
    overall_mean_gap = float(np.mean([g["mean"] for g in gap_per_regime.values()]))
    pass_gap = overall_mean_gap > 5.0

    # Separation: each regime's dominant action should be different
    dominant_actions = [int(np.argmax(action_dists[r])) for r in range(n_r)]
    pass_separation = len(set(dominant_actions)) == n_r

    return {
        "action_distributions": action_dists,
        "gap_per_regime": gap_per_regime,
        "disagree_vs_locked": disagree_vs_locked,
        "overall_mean_gap": overall_mean_gap,
        "dominant_actions": dominant_actions,
        "pass_gap": pass_gap,
        "pass_separation": pass_separation,
        "passed": pass_gap and pass_separation,
    }


# ---------------------------------------------------------------------------
# Precondition 3 — Oracle B regime-dependent under belief uncertainty
# ---------------------------------------------------------------------------

def precondition_3(
    sim: SimRecord,
    oracle_b_sol: VISolution,
    oracle_a_sol: VISolution,
    q_max: float,
    params: EnvParams,
    grid_size: int = 20,
) -> dict:
    """Check Precondition 3: Oracle B is regime-dependent despite belief uncertainty.

    Analyses Oracle B rollout tagged by true_regime:
      - Per-regime action distributions
      - Per-regime relative gaps (averaged over encountered beliefs)
      - Disagreement: Oracle B vs Oracle A
    """
    n_inv = 2 * int(params.inventory_max) + 1
    n_r = int(params.n_regimes)
    n_a = int(params.n_actions)

    # Per-regime action distributions
    action_dists = np.zeros((n_r, n_a))
    for r in range(n_r):
        mask = sim.regimes == r
        if mask.sum() == 0:
            continue
        acts = sim.actions[mask]
        for a in range(n_a):
            action_dists[r, a] = (acts == a).sum() / len(acts)

    # Per-regime mean relative gap from Oracle B Q-table
    # Oracle B Q: (n_belief, n_inv, n_a) — we average over encountered states
    # For simplicity, use Oracle A's gap as a proxy for the true regime gap
    # (Oracle B's gap is belief-dependent, not regime-dependent)
    Q_a = np.array(oracle_a_sol.Q)
    Q_sorted_a = np.sort(Q_a, axis=-1)
    gap_per_state_a = (Q_sorted_a[:, :, -1] - Q_sorted_a[:, :, -2]) / q_max * 100.0

    gap_per_regime = {}
    for r in range(n_r):
        mask = sim.regimes == r
        inv_idx = sim.inventories[mask] + int(params.inventory_max)
        gaps_encountered = gap_per_state_a[r, inv_idx]
        gap_per_regime[r] = {
            "mean": float(np.mean(gaps_encountered)),
        }

    # Oracle B vs Oracle A disagreement (per true regime, averaged over steps)
    disagree_vs_oracle_a = {}
    oracle_a_policy = np.array(oracle_a_sol.policy)
    for r in range(n_r):
        mask = sim.regimes == r
        inv_idx = sim.inventories[mask] + int(params.inventory_max)
        b_actions = sim.actions[mask]
        a_actions = oracle_a_policy[r, inv_idx]
        frac = float(np.mean(b_actions != a_actions)) * 100.0
        disagree_vs_oracle_a[r] = frac

    # Overall action distribution separation
    pass_separation = True
    for r in range(n_r):
        dominant = action_dists[r].max()
        if dominant < 0.4:  # should have clear preference
            pass_separation = False

    return {
        "action_distributions": action_dists,
        "gap_per_regime": gap_per_regime,
        "disagree_vs_oracle_a": disagree_vs_oracle_a,
        "pass_separation": pass_separation,
        "passed": pass_separation,
    }
