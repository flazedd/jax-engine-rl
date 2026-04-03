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
    """Load MC optimal actions from plots/mc_optimal.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "plots", "mc_optimal.json")
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

def compute_vi_policy(sim_config: SimConfig, rng_key, gamma=0.99,
                      n_mc_episodes=100):
    """Compute optimal mixed-regime policy via value iteration.

    State space: (regime, inventory) where inventory is discretised to
    integers in [-max_inv, +max_inv].

    Uses analytical reward decomposition to avoid sparse inventory binning:
      R(r, inv, a) = E[sc|r,a] + inv*E[Δmid|r,a]
                     - penalty*(inv² + 2*inv*E[Δinv|r,a] + E[Δinv²|r,a])
    where sc (spread capture), Δmid, Δinv are inventory-independent,
    so all MC samples per (regime, action) can be pooled.

    Returns (policy, values):
      policy : int array (N_REGIMES, n_inv) — optimal action per state
      values : float array (N_REGIMES, n_inv) — state values
    """
    max_inv = sim_config.max_inventory
    n_inv = 2 * max_inv + 1
    n_steps = sim_config.max_steps
    penalty = sim_config.inventory_penalty

    # Per (regime, action) statistics — pooled across all inventory levels
    mean_sc = np.zeros((N_REGIMES, N_ACTIONS))      # spread capture
    mean_dmid = np.zeros((N_REGIMES, N_ACTIONS))     # mid price change
    mean_dinv = np.zeros((N_REGIMES, N_ACTIONS))     # inventory change
    mean_dinv2 = np.zeros((N_REGIMES, N_ACTIONS))    # squared inv change
    # Δinv distribution for transition probabilities
    # Δinv is typically small: -1, 0, or +1 (agent_order_size=1)
    # but could be fractional fills; we'll bin integer Δinv
    max_dinv = 3  # allow Δinv in [-3, +3]
    n_dinv = 2 * max_dinv + 1
    dinv_hist = np.zeros((N_REGIMES, N_ACTIONS, n_dinv))

    # ── Step 1: MC estimation with analytical decomposition ──
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

            inv = np.array(outputs["inventory"])     # (n_ep, n_steps)
            rew = np.array(outputs["reward"])         # (n_ep, n_steps)
            mid = np.array(outputs["mid_price"])      # (n_ep, n_steps)
            done = np.array(outputs["done"])           # (n_ep, n_steps)

            # inv_before[t] and mid_before[t] = state entering step t
            inv_before = np.concatenate(
                [np.zeros((n_mc_episodes, 1)), inv[:, :-1]], axis=1)
            mid_before = np.concatenate(
                [np.full((n_mc_episodes, 1), 100.0), mid[:, :-1]], axis=1)

            # Valid = not done AND not at inventory boundary (where
            # clamping distorts Δinv, making it inventory-dependent)
            was_done = np.concatenate(
                [np.zeros((n_mc_episodes, 1), dtype=bool), done[:, :-1]],
                axis=1)
            unclamped = np.abs(inv_before) < max_inv
            valid = ~was_done & unclamped

            # Compute per-step quantities (inventory-independent components)
            d_inv = inv - inv_before                   # Δinv per step
            d_mid = mid - mid_before                   # Δmid per step
            new_inv = inv                              # inventory after step
            # spread_capture = reward + penalty*new_inv² - inv_before*Δmid
            # (undo the mtm and penalty terms to isolate spread capture)
            sc = rew + penalty * new_inv**2 - inv_before * d_mid

            # Pool all valid samples
            v_sc = sc[valid]
            v_dmid = d_mid[valid]
            v_dinv = d_inv[valid]

            n_valid = v_sc.shape[0]
            if n_valid > 0:
                mean_sc[regime, action_idx] = np.mean(v_sc)
                mean_dmid[regime, action_idx] = np.mean(v_dmid)
                mean_dinv[regime, action_idx] = np.mean(v_dinv)
                mean_dinv2[regime, action_idx] = np.mean(v_dinv**2)

                # Bin Δinv into integer buckets for transition probabilities
                dinv_rounded = np.clip(np.round(v_dinv).astype(int),
                                       -max_dinv, max_dinv)
                for d in range(-max_dinv, max_dinv + 1):
                    dinv_hist[regime, action_idx, d + max_dinv] = \
                        np.sum(dinv_rounded == d)
                dinv_hist[regime, action_idx] /= n_valid

            print(".", end="", flush=True)
    print(" done", flush=True)

    # ── Build reward and transition tables analytically ──
    inv_grid = np.arange(n_inv) - max_inv  # [-max_inv, ..., +max_inv]

    # R(r, inv, a) = E[sc|r,a] + inv*E[Δmid|r,a]
    #   - penalty * E[clip(inv+Δinv_raw, -max, +max)²|r,a]
    # The penalty term depends on inventory due to clamping, so we compute
    # it explicitly using the Δinv distribution.
    reward_table = np.zeros((N_REGIMES, n_inv, N_ACTIONS))
    for r in range(N_REGIMES):
        for a in range(N_ACTIONS):
            sc = mean_sc[r, a]
            dm = mean_dmid[r, a]
            for i_idx in range(n_inv):
                inv = inv_grid[i_idx]
                # E[clip(inv + Δinv, -max, max)²] from Δinv distribution
                expected_clipped_inv2 = 0.0
                for d in range(-max_dinv, max_dinv + 1):
                    prob = dinv_hist[r, a, d + max_dinv]
                    if prob > 0:
                        new_inv = np.clip(inv + d, -max_inv, max_inv)
                        expected_clipped_inv2 += prob * new_inv**2
                reward_table[r, i_idx, a] = (
                    sc + inv * dm - penalty * expected_clipped_inv2
                )

    # T[r, i, a, i'] = P(Δinv = i' - i | r, a)
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
                            # Clip: stay at boundary
                            trans_probs[r, i, a, np.clip(j, 0, n_inv - 1)] += prob

    # ── Step 2: Value iteration ──
    # Regime transitions BEFORE step dynamics (step.py:73-76), so reward
    # and inventory transition depend on the NEW regime r', not current r.
    # Bellman: Q(r,i,a) = sum_{r'} P(r'|r) * [R(r',i,a) + gamma * sum_{i'} P(i'|r',i,a) * V(r',i')]
    print("    Value iteration ... ", end="", flush=True)
    tm = np.array(TRANSITION_MATRIX)
    V = np.zeros((N_REGIMES, n_inv))
    policy = np.zeros((N_REGIMES, n_inv), dtype=int)

    for iteration in range(2000):
        Q = np.zeros((N_ACTIONS, N_REGIMES, n_inv))

        for a in range(N_ACTIONS):
            for r in range(N_REGIMES):
                q_ri = np.zeros(n_inv)
                for rp in range(N_REGIMES):
                    R = reward_table[rp, :, a]         # (n_inv,)
                    T = trans_probs[rp, :, a, :]       # (n_inv, n_inv)
                    q_ri += tm[r, rp] * (R + gamma * (T @ V[rp]))
                Q[a, r] = q_ri

        V_new = np.max(Q, axis=0)
        new_policy = np.argmax(Q, axis=0)

        delta = np.max(np.abs(V_new - V))
        if delta < 1e-6:
            print(f"converged at iteration {iteration}", flush=True)
            break
        V = V_new
        policy = new_policy
    else:
        print(f"max iterations (delta={delta:.2e})", flush=True)

    return policy, V


def save_vi_policy(policy, values, path=None):
    """Save VI policy and values to JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "plots", "vi_optimal.json")
        path = os.path.normpath(path)
    data = {
        "policy": policy.tolist(),
        "values": values.tolist(),
        "max_inventory": (policy.shape[1] - 1) // 2,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Saved VI policy to {path}")


def load_vi_policy(path=None):
    """Load VI policy from JSON. Returns (policy, values) as numpy arrays."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "plots", "vi_optimal.json")
        path = os.path.normpath(path)
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
                       n_episodes=100):
    """Evaluate the VI-optimal oracle on mixed (free-transition) regimes.

    At each step the oracle reads the true regime and current inventory
    to look up the optimal action from the VI policy table.

    Returns dict: 'mean_reward', 'std_reward', 'mean_episode_length', 'rewards'
    """
    max_inv = sim_config.max_inventory
    n_inv = 2 * max_inv + 1
    n_steps = sim_config.max_steps

    step_fn = make_step_fn(sim_config, locked_regime=-1)
    policy_jax = jnp.array(policy, dtype=jnp.int32)

    def run_one(key):
        k_init, _ = jax.random.split(key)
        sim_state = init_state(sim_config, k_init)

        def step(carry, _):
            sim_state, total_reward, ep_len = carry
            inv_idx = jnp.clip(
                jnp.round(sim_state.inventory).astype(jnp.int32) + max_inv,
                0, n_inv - 1)
            action = policy_jax[sim_state.regime, inv_idx]
            new_state, out = step_fn(sim_state, action)
            total_reward = total_reward + out["reward"]
            ep_len = ep_len + jnp.where(sim_state.done, 0, 1)
            return (new_state, total_reward, ep_len), None

        init_carry = (sim_state, jnp.float32(0.0), jnp.int32(0))
        (_, total_reward, ep_len), _ = jax.lax.scan(
            step, init_carry, None, length=n_steps)
        return total_reward, ep_len

    keys = jax.random.split(rng_key, n_episodes)
    rewards, lengths = jax.vmap(run_one)(keys)

    return {
        "mean_reward": jnp.mean(rewards),
        "std_reward": jnp.std(rewards),
        "mean_episode_length": jnp.mean(lengths.astype(jnp.float32)),
        "rewards": rewards,
    }
