"""Oracle agents for upper-bound evaluation.

1. Myopic oracle: plays MC-optimal action per true regime (ignores inventory).
2. VI oracle: value-iteration-optimal over (regime, inventory) state space.
   Accounts for regime transition risk and inventory carry-over.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from lob_sim.config import SimConfig
from lob_sim.regime import N_REGIMES, TRANSITION_MATRIX
from lob_sim.actions import N_ACTIONS, ACTION_TABLE
from lob_sim.state import init_state
from lob_sim.step import make_step_fn, run_episode


# ── Myopic oracle (existing) ────────────────────────────────────

def load_mc_optimal() -> list[int]:
    """Load MC optimal actions from results/mc_optimal.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "results", "mc_optimal.json")
    path = os.path.normpath(path)
    with open(path) as f:
        data = json.load(f)
    return data["optimal_actions"]


def evaluate_oracle(sim_config: SimConfig, rng_key, n_episodes=50,
                    locked_regime=-1, mc_optimal=None):
    """Run the myopic oracle (MC-optimal action per true regime).

    Returns dict: 'mean_reward', 'std_reward', 'mean_episode_length', 'rewards'
    """
    if mc_optimal is None:
        mc_optimal = load_mc_optimal()

    optimal_table = jnp.array(mc_optimal, dtype=jnp.int32)
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)
    n_steps = sim_config.max_steps

    def run_one_episode(key):
        k_init, k_run = jax.random.split(key)
        sim_state = init_state(sim_config, k_init)

        def step(carry, _):
            sim_state, total_reward, ep_len = carry
            action = optimal_table[sim_state.regime]
            new_sim_state, sim_out = step_fn(sim_state, action)
            total_reward = total_reward + sim_out["reward"]
            ep_len = ep_len + jnp.where(sim_state.done, 0, 1)
            return (new_sim_state, total_reward, ep_len), None

        init_carry = (sim_state, jnp.float32(0.0), jnp.int32(0))
        (_, total_reward, ep_len), _ = jax.lax.scan(
            step, init_carry, None, length=n_steps)
        return total_reward, ep_len

    keys = jax.random.split(rng_key, n_episodes)
    rewards, lengths = jax.vmap(run_one_episode)(keys)

    return {
        "mean_reward": jnp.mean(rewards),
        "std_reward": jnp.std(rewards),
        "mean_episode_length": jnp.mean(lengths.astype(jnp.float32)),
        "rewards": rewards,
    }


# ── VI oracle (inventory-aware) ─────────────────────────────────

def _mc_estimate(sim_config, rng_key, n_mc_episodes, horizon=30):
    """Collect per-(regime, inventory, action) MC statistics for VI.

    Uses multi-step rollouts (same methodology as reward_landscape.py):
    at each (regime, inv, action), runs n_mc_episodes trials of HORIZON steps
    spamming the same action, and records the mean per-step reward and the
    inventory transition over the full horizon.

    Returns (reward_table, trans_probs) that can be reused across different
    transition matrices.
    """
    max_inv = sim_config.max_inventory
    n_inv = 2 * max_inv + 1
    inv_grid = np.arange(n_inv) - max_inv

    reward_table = np.zeros((N_REGIMES, n_inv, N_ACTIONS))
    trans_probs = np.zeros((N_REGIMES, n_inv, N_ACTIONS, n_inv))

    print(f"    MC estimation (horizon={horizon}) ", end="", flush=True)
    for regime in range(N_REGIMES):
        step_fn = make_step_fn(sim_config, locked_regime=regime)

        def _multi_step(inv_f32, action_i32, rng_key):
            state = init_state(sim_config, rng_key)
            state = state._replace(inventory=inv_f32)
            actions = jnp.full((horizon,), action_i32, dtype=jnp.int32)

            def step(s, a):
                new_s, out = step_fn(s, a)
                return new_s, out["reward"]

            final_state, rewards = jax.lax.scan(step, state, actions)
            return rewards.mean(), final_state.inventory

        batched = jax.jit(jax.vmap(_multi_step, in_axes=(None, None, 0)))

        for action_idx in range(N_ACTIONS):
            for i_idx, inv in enumerate(inv_grid):
                rng_key, k = jax.random.split(rng_key)
                keys = jax.random.split(k, n_mc_episodes)
                mean_rewards, final_invs = batched(
                    jnp.float32(inv), jnp.int32(action_idx), keys)

                reward_table[regime, i_idx, action_idx] = float(
                    jnp.mean(mean_rewards))

                # Inventory transition: where does the agent end up after
                # HORIZON steps? Bin into inventory grid.
                final_inv_idx = np.clip(
                    np.round(np.array(final_invs)).astype(int) + max_inv,
                    0, n_inv - 1)
                for j in range(n_inv):
                    count = np.sum(final_inv_idx == j)
                    if count > 0:
                        trans_probs[regime, i_idx, action_idx, j] = (
                            count / n_mc_episodes)

            print(".", end="", flush=True)
    print(" done", flush=True)

    return reward_table, trans_probs


def _vi_isolated(reward_table, trans_probs, regime, gamma=0.99, verbose=True):
    """Value iteration for a single isolated regime.

    Solves a 1D MDP over inventory only (no regime transitions).
    Uses Gauss-Seidel (in-place) updates for faster convergence.

    Args:
        reward_table: (N_REGIMES, n_inv, N_ACTIONS)
        trans_probs: (N_REGIMES, n_inv, N_ACTIONS, n_inv)
        regime: int — which regime to solve for
        gamma: discount factor

    Returns (policy, values, Q, info)
        policy: (n_inv,) int — optimal action per inventory
        values: (n_inv,) float — state values
        Q: (n_inv, N_ACTIONS) float — Q-values
    """
    r = regime
    n_inv = reward_table.shape[1]
    R = reward_table[r]     # (n_inv, N_ACTIONS)
    T = trans_probs[r]      # (n_inv, N_ACTIONS, n_inv)

    V = np.zeros(n_inv)
    policy = np.zeros(n_inv, dtype=int)
    deltas = []
    policy_stable_since = 0

    for iteration in range(5000):
        max_delta = 0.0
        for i in range(n_inv):
            q_vals = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                q_vals[a] = R[i, a] + gamma * np.dot(T[i, a, :], V)
            v_new = np.max(q_vals)
            max_delta = max(max_delta, abs(v_new - V[i]))
            V[i] = v_new  # Gauss-Seidel: update in-place

        policy_new = np.zeros(n_inv, dtype=int)
        Q = np.zeros((n_inv, N_ACTIONS))
        for i in range(n_inv):
            for a in range(N_ACTIONS):
                Q[i, a] = R[i, a] + gamma * np.dot(T[i, a, :], V)
            policy_new[i] = np.argmax(Q[i])

        deltas.append(float(max_delta))
        if not np.array_equal(policy_new, policy):
            policy_stable_since = iteration
        policy = policy_new

        if max_delta < 1e-8:
            if verbose:
                print(f"converged at iter {iteration} "
                      f"(policy stable since {policy_stable_since})",
                      flush=True)
            break
    else:
        if verbose:
            print(f"max iterations (delta={max_delta:.2e})", flush=True)

    # Compute Bellman residual: max |V - max_a Q(s,a)|
    bellman_residual = max(abs(V[i] - np.max(Q[i])) for i in range(n_inv))

    info = {
        "n_iterations": min(iteration + 1, 5000),
        "final_delta": deltas[-1],
        "policy_stable_since": policy_stable_since,
        "max_bellman_residual": bellman_residual,
        "deltas": deltas,
    }
    return policy, V, Q, info


def _vi_mixed(reward_table, trans_probs, transition_matrix, gamma=0.99,
              verbose=True):
    """Value iteration under regime switching (mixed).

    Full (regime, inventory) state space with Gauss-Seidel updates.

    Returns (policy, values, Q, info)
        policy: (N_REGIMES, n_inv) int
        values: (N_REGIMES, n_inv) float
        Q: (N_REGIMES, n_inv, N_ACTIONS) float
    """
    n_inv = reward_table.shape[1]
    tm = np.array(transition_matrix)

    V = np.zeros((N_REGIMES, n_inv))
    policy = np.zeros((N_REGIMES, n_inv), dtype=int)
    deltas = []
    policy_stable_since = 0

    for iteration in range(5000):
        max_delta = 0.0
        for r in range(N_REGIMES):
            for i in range(n_inv):
                q_vals = np.zeros(N_ACTIONS)
                for a in range(N_ACTIONS):
                    for rp in range(N_REGIMES):
                        q_vals[a] += tm[r, rp] * (
                            reward_table[rp, i, a]
                            + gamma * np.dot(trans_probs[rp, i, a, :], V[rp]))
                v_new = np.max(q_vals)
                max_delta = max(max_delta, abs(v_new - V[r, i]))
                V[r, i] = v_new

        # Recompute Q and policy from converged V
        Q = np.zeros((N_REGIMES, n_inv, N_ACTIONS))
        policy_new = np.zeros((N_REGIMES, n_inv), dtype=int)
        for r in range(N_REGIMES):
            for i in range(n_inv):
                for a in range(N_ACTIONS):
                    for rp in range(N_REGIMES):
                        Q[r, i, a] += tm[r, rp] * (
                            reward_table[rp, i, a]
                            + gamma * np.dot(trans_probs[rp, i, a, :], V[rp]))
                policy_new[r, i] = np.argmax(Q[r, i])

        deltas.append(float(max_delta))
        if not np.array_equal(policy_new, policy):
            policy_stable_since = iteration
        policy = policy_new

        if max_delta < 1e-8:
            if verbose:
                print(f"converged at iter {iteration} "
                      f"(policy stable since {policy_stable_since})",
                      flush=True)
            break
    else:
        if verbose:
            print(f"max iterations (delta={max_delta:.2e})", flush=True)

    bellman_residual = 0.0
    for r in range(N_REGIMES):
        for i in range(n_inv):
            bellman_residual = max(bellman_residual,
                                  abs(V[r, i] - np.max(Q[r, i])))

    info = {
        "n_iterations": min(iteration + 1, 5000),
        "final_delta": deltas[-1],
        "policy_stable_since": policy_stable_since,
        "max_bellman_residual": bellman_residual,
        "deltas": deltas,
    }
    return policy, V, Q, info


def compute_vi_isolated_and_mixed(sim_config: SimConfig, rng_key, gamma=0.99,
                                  n_mc_episodes=4000, horizon=30):
    """Compute 4 optimal policies: 3 isolated (one per regime) + 1 mixed.

    Uses multi-step MC rollouts (horizon steps per trial) matching
    reward_landscape.py methodology. Runs two seeds and reports
    disagreements. Uses seed 1 for the final policies.

    Returns:
        iso_results: list of 3 tuples (policy, values, Q, info)
            policy: (n_inv,) int — optimal action per inventory for this regime
            Q: (n_inv, N_ACTIONS) float
        mix_result: (policy, values, Q, info)
            policy: (N_REGIMES, n_inv) int
            Q: (N_REGIMES, n_inv, N_ACTIONS) float
    """
    max_inv = sim_config.max_inventory
    n_inv = 2 * max_inv + 1
    display_lo = 5
    display_hi = n_inv - 5
    regime_names = ["Noise", "Bull", "Bear"]

    k1, k2 = jax.random.split(rng_key)

    print(f"\n  Seed 1 (n_mc={n_mc_episodes}, horizon={horizon}):")
    rt1, tp1 = _mc_estimate(sim_config, k1, n_mc_episodes, horizon=horizon)

    print(f"  Seed 2 (n_mc={n_mc_episodes}, horizon={horizon}):")
    rt2, tp2 = _mc_estimate(sim_config, k2, n_mc_episodes, horizon=horizon)

    # ── Compute 3 isolated policies from seed 1 ──
    iso_results = []
    for r in range(N_REGIMES):
        print(f"  VI isolated {regime_names[r]} (seed 1) ... ", end="",
              flush=True)
        policy, V, Q, info = _vi_isolated(rt1, tp1, r, gamma)
        iso_results.append((policy, V, Q, info))

    # ── Compute mixed policy from seed 1 ──
    print("  VI mixed (seed 1) ... ", end="", flush=True)
    mix_policy, mix_V, mix_Q, mix_info = _vi_mixed(
        rt1, tp1, TRANSITION_MATRIX, gamma)

    # ── Seed 2 for convergence check ──
    iso2_policies = []
    for r in range(N_REGIMES):
        p2, _, _, _ = _vi_isolated(rt2, tp2, r, gamma, verbose=False)
        iso2_policies.append(p2)

    mix2_policy, _, _, _ = _vi_mixed(rt2, tp2, TRANSITION_MATRIX, gamma,
                                      verbose=False)

    # ── Report disagreements ──
    print(f"\n  Convergence check (excluding boundary ±5):")
    for r in range(N_REGIMES):
        p1 = iso_results[r][0][display_lo:display_hi]
        p2 = iso2_policies[r][display_lo:display_hi]
        diff = int(np.sum(p1 != p2))
        status = "CONVERGED" if diff == 0 else "NOT CONVERGED"
        print(f"    Isolated {regime_names[r]:>5s}: {diff} disagreements "
              f"({status})")
        iso_results[r][3]["mc_converged"] = diff == 0
        iso_results[r][3]["mc_seed_disagreements"] = diff

    mix_diff = int(np.sum(
        mix_policy[:, display_lo:display_hi]
        != mix2_policy[:, display_lo:display_hi]))
    print(f"    Mixed:           {mix_diff} disagreements "
          f"({'CONVERGED' if mix_diff == 0 else 'NOT CONVERGED'})")
    mix_info["mc_converged"] = mix_diff == 0
    mix_info["mc_seed_disagreements"] = mix_diff

    # ── Assemble iso_policy in (N_REGIMES, n_inv) shape for backward compat ──
    iso_policy = np.stack([iso_results[r][0] for r in range(N_REGIMES)])
    iso_values = np.stack([iso_results[r][1] for r in range(N_REGIMES)])
    iso_Q = np.stack([iso_results[r][2] for r in range(N_REGIMES)])

    # Merge info from all 3 isolated runs
    iso_info = {
        "Q": iso_Q,
        "per_regime": [iso_results[r][3] for r in range(N_REGIMES)],
        # Summary: worst-case convergence stats
        "n_iterations": max(iso_results[r][3]["n_iterations"]
                           for r in range(N_REGIMES)),
        "final_delta": max(iso_results[r][3]["final_delta"]
                          for r in range(N_REGIMES)),
        "policy_stable_since": max(iso_results[r][3]["policy_stable_since"]
                                   for r in range(N_REGIMES)),
        "max_bellman_residual": max(iso_results[r][3].get(
            "max_bellman_residual", iso_results[r][3]["final_delta"])
            for r in range(N_REGIMES)),
        "deltas": iso_results[0][3]["deltas"],  # just use noise for plot
    }

    mix_info["Q"] = mix_Q

    return ((iso_policy, iso_values, iso_info),
            (mix_policy, mix_V, mix_info))


def _results_path(filename):
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "results", filename))


def save_vi_policy(policy, values, path=None):
    """Save VI policy and values to JSON."""
    if path is None:
        path = _results_path("vi_optimal.json")
    data = {
        "policy": policy.tolist(),
        "values": values.tolist(),
        "max_inventory": (policy.shape[1] - 1) // 2,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Saved VI policy to {path}")


def save_vi_isolated(policy, values, path=None):
    """Save isolated-regime VI policy and values to JSON."""
    if path is None:
        path = _results_path("vi_isolated.json")
    data = {
        "policy": policy.tolist(),
        "values": values.tolist(),
        "max_inventory": (policy.shape[1] - 1) // 2,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Saved isolated VI policy to {path}")


def load_vi_policy(path=None):
    """Load VI policy from JSON. Returns (policy, values) as numpy arrays."""
    if path is None:
        path = _results_path("vi_optimal.json")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["policy"], dtype=int), np.array(data["values"])


def load_vi_isolated(path=None):
    """Load isolated-regime VI policy. Returns (policy, values) as numpy arrays."""
    if path is None:
        path = _results_path("vi_isolated.json")
    with open(path) as f:
        data = json.load(f)
    return np.array(data["policy"], dtype=int), np.array(data["values"])


def print_vi_policy(policy, max_inv=20):
    """Pretty-print the VI policy table at key inventory levels."""
    regime_names = ["Noise", "Bull ", "Bear "]
    inv_levels = [-15, -10, -5, 0, 5, 10, 15]
    at = np.array(ACTION_TABLE)

    header = "         " + "".join(f"{'inv='+str(i):>10s}" for i in inv_levels)
    print(header)
    for r in range(N_REGIMES):
        parts = []
        for inv in inv_levels:
            idx = inv + max_inv
            a = policy[r, idx]
            bid, ask = int(at[a][0]), int(at[a][1])
            parts.append(f"({bid},{ask})")
        row = "    " + regime_names[r] + "  " + "".join(f"{p:>10s}" for p in parts)
        print(row)


def evaluate_vi_oracle(sim_config: SimConfig, rng_key, policy,
                       n_episodes=100, locked_regime=-1,
                       return_trajectories=False):
    """Evaluate the VI-optimal oracle.

    At each step the oracle reads the true regime and current inventory
    to look up the optimal action from the VI policy table.

    If return_trajectories=True, also returns per-step rewards and actions
    for cumulative reward curves and action-inventory analysis.

    Returns dict: 'mean_reward', 'std_reward', 'mean_episode_length',
                  'rewards', and optionally 'step_rewards', 'step_actions',
                  'step_inventories', 'step_mask'.
    """
    max_inv = sim_config.max_inventory
    n_inv = 2 * max_inv + 1
    n_steps = sim_config.max_steps

    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)
    policy_jax = jnp.array(policy, dtype=jnp.int32)

    def run_one(key):
        k_init, _ = jax.random.split(key)
        sim_state = init_state(sim_config, k_init)

        def step(carry, _):
            sim_state, total_reward, ep_len = carry
            was_live = ~sim_state.done
            inv_idx = jnp.clip(
                jnp.round(sim_state.inventory).astype(jnp.int32) + max_inv,
                0, n_inv - 1)
            action = policy_jax[sim_state.regime, inv_idx]
            new_state, out = step_fn(sim_state, action)
            reward = out["reward"]
            total_reward = total_reward + reward
            ep_len = ep_len + jnp.where(sim_state.done, 0, 1)
            return (new_state, total_reward, ep_len), (reward, action,
                                                        sim_state.inventory,
                                                        was_live)

        init_carry = (sim_state, jnp.float32(0.0), jnp.int32(0))
        (_, total_reward, ep_len), step_data = jax.lax.scan(
            step, init_carry, None, length=n_steps)
        return total_reward, ep_len, step_data

    keys = jax.random.split(rng_key, n_episodes)
    rewards, lengths, step_data = jax.vmap(run_one)(keys)
    # step_data: each element is (n_episodes, n_steps)

    result = {
        "mean_reward": jnp.mean(rewards),
        "std_reward": jnp.std(rewards),
        "mean_episode_length": jnp.mean(lengths.astype(jnp.float32)),
        "rewards": rewards,
    }

    if return_trajectories:
        step_rewards, step_actions, step_inventories, step_mask = step_data
        result["step_rewards"] = step_rewards
        result["step_actions"] = step_actions
        result["step_inventories"] = step_inventories
        result["step_mask"] = step_mask

    return result
