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

def _mc_estimate(sim_config, rng_key, n_mc_episodes):
    """Collect per-(regime, action) MC statistics for VI.

    Reward = spread_capture + inventory * Δmid (no penalty term).
    We decompose into inventory-independent spread capture and Δmid,
    then reconstruct R[r, i, a] = sc[r,a] + i * Δmid[r,a].

    Returns (reward_table, trans_probs) that can be reused across different
    transition matrices.
    """
    max_inv = sim_config.max_inventory
    n_inv = 2 * max_inv + 1
    n_steps = sim_config.max_steps

    mean_sc = np.zeros((N_REGIMES, N_ACTIONS))
    mean_dmid = np.zeros((N_REGIMES, N_ACTIONS))
    max_dinv = 3
    n_dinv = 2 * max_dinv + 1
    dinv_hist = np.zeros((N_REGIMES, N_ACTIONS, n_dinv))

    print("    MC estimation ", end="", flush=True)
    for regime in range(N_REGIMES):
        rng_key, k_regime = jax.random.split(rng_key)
        keys = jax.random.split(k_regime, n_mc_episodes)

        def _run_one(key, actions_arr):
            return run_episode(sim_config, key, actions_arr,
                               locked_regime=regime)
        vmapped = jax.vmap(_run_one, in_axes=(0, None))

        for action_idx in range(N_ACTIONS):
            actions_arr = jnp.full(n_steps, action_idx, dtype=jnp.int32)
            _, outputs = vmapped(keys, actions_arr)

            inv = np.array(outputs["inventory"])
            rew = np.array(outputs["reward"])
            mid = np.array(outputs["mid_price"])
            done = np.array(outputs["done"])

            inv_before = np.concatenate(
                [np.zeros((n_mc_episodes, 1)), inv[:, :-1]], axis=1)
            mid_before = np.concatenate(
                [np.full((n_mc_episodes, 1), 100.0), mid[:, :-1]], axis=1)

            was_done = np.concatenate(
                [np.zeros((n_mc_episodes, 1), dtype=bool), done[:, :-1]],
                axis=1)
            unclamped = np.abs(inv_before) < max_inv
            valid = ~was_done & unclamped

            d_inv = inv - inv_before
            d_mid = mid - mid_before
            # Extract spread capture: reward - inventory * Δmid
            sc = rew - inv_before * d_mid

            v_sc = sc[valid]
            v_dmid = d_mid[valid]
            v_dinv = d_inv[valid]

            n_valid = v_sc.shape[0]
            if n_valid > 0:
                mean_sc[regime, action_idx] = np.mean(v_sc)
                mean_dmid[regime, action_idx] = np.mean(v_dmid)

                dinv_rounded = np.clip(np.round(v_dinv).astype(int),
                                       -max_dinv, max_dinv)
                for d in range(-max_dinv, max_dinv + 1):
                    dinv_hist[regime, action_idx, d + max_dinv] = \
                        np.sum(dinv_rounded == d)
                dinv_hist[regime, action_idx] /= n_valid

            print(".", end="", flush=True)
    print(" done", flush=True)

    # ── Build reward and transition tables ──
    inv_grid = np.arange(n_inv) - max_inv

    reward_table = np.zeros((N_REGIMES, n_inv, N_ACTIONS))
    for r in range(N_REGIMES):
        for a in range(N_ACTIONS):
            for i_idx in range(n_inv):
                reward_table[r, i_idx, a] = (
                    mean_sc[r, a] + inv_grid[i_idx] * mean_dmid[r, a]
                )

    trans_probs = np.zeros((N_REGIMES, n_inv, N_ACTIONS, n_inv))
    for r in range(N_REGIMES):
        for a in range(N_ACTIONS):
            for d in range(-max_dinv, max_dinv + 1):
                prob = dinv_hist[r, a, d + max_dinv]
                if prob > 0:
                    for i in range(n_inv):
                        j = i + d
                        if 0 <= j < n_inv:
                            trans_probs[r, i, a, j] += prob
                        else:
                            trans_probs[r, i, a, np.clip(j, 0, n_inv - 1)] += prob

    return reward_table, trans_probs


def _value_iteration(reward_table, trans_probs, transition_matrix, gamma=0.99,
                     verbose=True):
    """Run value iteration given pre-computed reward/transition tables.

    Args:
        reward_table: (N_REGIMES, n_inv, N_ACTIONS)
        trans_probs: (N_REGIMES, n_inv, N_ACTIONS, n_inv)
        transition_matrix: (N_REGIMES, N_REGIMES) — row-stochastic
        gamma: discount factor
        verbose: print convergence info

    Returns (policy, values, info) where info contains convergence data.
    """
    n_inv = reward_table.shape[1]
    tm = np.array(transition_matrix)

    V = np.zeros((N_REGIMES, n_inv))
    policy = np.zeros((N_REGIMES, n_inv), dtype=int)
    deltas = []
    policy_stable_since = 0

    for iteration in range(2000):
        Q = np.zeros((N_ACTIONS, N_REGIMES, n_inv))

        for a in range(N_ACTIONS):
            for r in range(N_REGIMES):
                q_ri = np.zeros(n_inv)
                for rp in range(N_REGIMES):
                    R = reward_table[rp, :, a]
                    T = trans_probs[rp, :, a, :]
                    q_ri += tm[r, rp] * (R + gamma * (T @ V[rp]))
                Q[a, r] = q_ri

        V_new = np.max(Q, axis=0)
        new_policy = np.argmax(Q, axis=0)

        delta = np.max(np.abs(V_new - V))
        deltas.append(float(delta))

        if not np.array_equal(new_policy, policy):
            policy_stable_since = iteration

        if delta < 1e-6:
            if verbose:
                print(f"converged at iteration {iteration} "
                      f"(policy stable since {policy_stable_since})", flush=True)
            break
        V = V_new
        policy = new_policy
    else:
        if verbose:
            print(f"max iterations (delta={delta:.2e})", flush=True)

    # Compute Bellman residual for final policy as sanity check
    bellman_residual = np.zeros((N_REGIMES, n_inv))
    for r in range(N_REGIMES):
        for i in range(n_inv):
            a = policy[r, i]
            q = 0.0
            for rp in range(N_REGIMES):
                R = reward_table[rp, i, a]
                future = gamma * np.dot(trans_probs[rp, i, a, :], V[rp])
                q += tm[r, rp] * (R + future)
            bellman_residual[r, i] = abs(q - V[r, i])

    info = {
        "n_iterations": min(iteration + 1, 2000),
        "final_delta": deltas[-1],
        "policy_stable_since": policy_stable_since,
        "max_bellman_residual": float(np.max(bellman_residual)),
        "deltas": deltas,
    }

    return policy, V, info


def compute_vi_policy(sim_config: SimConfig, rng_key, gamma=0.99,
                      n_mc_episodes=100, transition_matrix=None):
    """Compute optimal policy via value iteration over (regime, inventory).

    Args:
        transition_matrix: optional override. Defaults to TRANSITION_MATRIX.
            Use np.eye(N_REGIMES) for isolated-regime policies.

    Returns (policy, values):
      policy : int array (N_REGIMES, n_inv) — optimal action per state
      values : float array (N_REGIMES, n_inv) — state values
    """
    if transition_matrix is None:
        transition_matrix = TRANSITION_MATRIX

    reward_table, trans_probs = _mc_estimate(sim_config, rng_key,
                                             n_mc_episodes)

    print("    Value iteration ... ", end="", flush=True)
    policy, V, info = _value_iteration(reward_table, trans_probs,
                                       transition_matrix, gamma)
    return policy, V, info


def compute_vi_isolated_and_mixed(sim_config: SimConfig, rng_key, gamma=0.99,
                                  n_mc_episodes=100, max_doublings=4):
    """Compute both isolated-regime and mixed-regime VI policies.

    Verifies MC convergence: runs estimation with two independent seeds
    and checks that policies agree. If not, doubles n_mc and retries
    (up to max_doublings times).

    Returns:
        isolated: (policy, values, info) with np.eye(3) transition matrix
        mixed: (policy, values, info) with TRANSITION_MATRIX
    """
    n_mc = n_mc_episodes
    max_inv = sim_config.max_inventory

    # Display range for convergence reporting (skip boundary)
    display_lo = 5
    display_hi = 2 * max_inv + 1 - 5

    for attempt in range(max_doublings + 1):
        k1, k2, rng_key = jax.random.split(rng_key, 3)

        print(f"\n  MC convergence check (n_mc={n_mc}, attempt {attempt+1})")

        print(f"    Seed A:")
        rt_a, tp_a = _mc_estimate(sim_config, k1, n_mc)
        print(f"    Seed B:")
        rt_b, tp_b = _mc_estimate(sim_config, k2, n_mc)

        # Average the two estimates for the final tables
        reward_table = (rt_a + rt_b) / 2
        trans_probs = (tp_a + tp_b) / 2

        # Compute policies from each seed independently
        iso_a, _, _ = _value_iteration(rt_a, tp_a, np.eye(N_REGIMES), gamma,
                                       verbose=False)
        iso_b, _, _ = _value_iteration(rt_b, tp_b, np.eye(N_REGIMES), gamma,
                                       verbose=False)
        mix_a, _, _ = _value_iteration(rt_a, tp_a, TRANSITION_MATRIX, gamma,
                                       verbose=False)
        mix_b, _, _ = _value_iteration(rt_b, tp_b, TRANSITION_MATRIX, gamma,
                                       verbose=False)

        # Check agreement in display range (skip boundary artifacts)
        n_cells = iso_a[:, display_lo:display_hi].size
        # Allow up to max_disagree cells to differ — these are at transition
        # points where two actions have near-identical Q-values.
        max_disagree = max(1, n_cells // 30)  # ~3% tolerance

        iso_diff = int(np.sum(iso_a[:, display_lo:display_hi]
                              != iso_b[:, display_lo:display_hi]))
        mix_diff = int(np.sum(mix_a[:, display_lo:display_hi]
                              != mix_b[:, display_lo:display_hi]))

        iso_ok = iso_diff <= max_disagree
        mix_ok = mix_diff <= max_disagree

        print(f"    Isolated policy: {n_cells - iso_diff}/{n_cells} agree "
              f"({'CONVERGED' if iso_ok else f'{iso_diff} differ'})")
        print(f"    Mixed policy:    {n_cells - mix_diff}/{n_cells} agree "
              f"({'CONVERGED' if mix_ok else f'{mix_diff} differ'})")
        print(f"    (tolerance: {max_disagree} cells)")

        if iso_ok and mix_ok:
            break

        if attempt < max_doublings:
            n_mc *= 2
            print(f"    Doubling to n_mc={n_mc}...")
        else:
            print(f"    WARNING: MC not fully converged after {max_doublings} "
                  f"doublings (n_mc={n_mc}). Using averaged estimates.")

    # Final VI from averaged tables
    print(f"\n    Final VI from averaged MC estimates (n_mc={n_mc} × 2 seeds):")
    print("    VI (isolated regimes) ... ", end="", flush=True)
    iso_policy, iso_values, iso_info = _value_iteration(
        reward_table, trans_probs, np.eye(N_REGIMES), gamma)

    print("    VI (mixed regimes) ... ", end="", flush=True)
    mix_policy, mix_values, mix_info = _value_iteration(
        reward_table, trans_probs, TRANSITION_MATRIX, gamma)

    # Store MC convergence metadata
    iso_info["n_mc_final"] = n_mc
    iso_info["mc_converged"] = bool(iso_ok)
    iso_info["mc_seed_disagreements"] = iso_diff
    mix_info["n_mc_final"] = n_mc
    mix_info["mc_converged"] = bool(mix_ok)
    mix_info["mc_seed_disagreements"] = mix_diff

    return (iso_policy, iso_values, iso_info), (mix_policy, mix_values, mix_info)


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
