"""Posterior-quality probe for meta-RL methods on MM E_final.

Given a trained checkpoint, run evaluation rollouts on the env, record
the agent's internal belief representation (GRU hidden for RL², posterior
μ for VariBAD), and the env's true regime label per timestep. Train a
classifier from belief → regime; report mean test accuracy and per-
timestep accuracy curves.

Spec lives at `results/milestones/M5/STEP5_PROBE_DESIGN.md`.

Currently MM-specific: assumes the env is `MMReducedEnv` (possibly
wrapped) and that `info` from each step contains `regime`, `bid_fill`,
`ask_fill`. Generalisation to non-MM envs is left as a follow-on if
needed.
"""
from __future__ import annotations

import json
import pickle
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@contextmanager
def _suppress_blas_matmul_warnings():
    """Suppress known-false-positive overflow/divide-by-zero RuntimeWarnings
    from `@` matmul on numpy 2.1 + OpenBLAS for certain array sizes. The
    matmul result is mathematically correct; these are spurious FPE flags
    raised by the vectorized BLAS code path. See numpy issue #27537 (and
    related). Sklearn's L-BFGS internals use `@`, so we wrap the calls.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        yield

from beliefs.hmm_posterior import full_update, initial_belief
from envs.mm_reduced import MMReducedEnv


# ---------------------------------------------------------------------------
# Inner-env extraction (walk through wrappers to find MMReducedEnv)
# ---------------------------------------------------------------------------


def get_inner_mm(env: Any) -> MMReducedEnv:
    """Walk down wrapper chain until we find the MMReducedEnv."""
    cur = env
    while not isinstance(cur, MMReducedEnv):
        if not hasattr(cur, "inner"):
            raise TypeError(
                f"could not find MMReducedEnv inside {type(cur).__name__} "
                "(no `inner` attribute)"
            )
        cur = cur.inner
    return cur


# ---------------------------------------------------------------------------
# Probe rollout collection
# ---------------------------------------------------------------------------


def collect_probe_rollouts(
    env: Any,
    agent: Any,
    agent_state: chex.ArrayTree,
    n_rollouts: int,
    rollout_length: int,
    key: chex.PRNGKey,
) -> dict[str, np.ndarray]:
    """Run `n_rollouts` parallel evaluation rollouts.

    Records per-timestep arrays shaped `[T, N, ...]`:
      - obs: agent observation
      - belief: agent's internal belief (carry into act for RL²,
        μ from extras for VariBAD)
      - regime: env's true regime (from info["regime"])
      - bid_fill, ask_fill, q: needed for analytical-posterior reference
      - action, reward, done

    Also computes per-step analytical_belief by running the HMM filter
    inline on the same trajectory.

    Returns dict of `np.ndarray`s ready for classifier input.
    """
    inner_mm = get_inner_mm(env)
    n_regimes = max(1, inner_mm.n_regimes)
    if n_regimes < 2:
        raise ValueError("probe requires n_regimes ≥ 2; got " + str(n_regimes))

    # vmapped env reset/step
    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)

    reset_keys = jax.random.split(key, n_rollouts + 1)
    keys_init, key = reset_keys[:n_rollouts], reset_keys[-1]
    env_states, obses = reset_v(keys_init)

    # Agent's recurrent carry (zeros at episode start)
    carry = agent.init_carry(n_rollouts)
    belief_key = getattr(agent, "belief_key", None)
    if belief_key is None:
        raise ValueError(
            "agent has no `belief_key` attribute; cannot identify belief vector"
        )

    # Initial analytical belief
    init_b = initial_belief(inner_mm)  # [n_regimes]
    analytical_belief = jnp.broadcast_to(init_b, (n_rollouts, n_regimes))

    out_obs = []
    out_belief = []
    out_regime = []
    out_action = []
    out_reward = []
    out_done = []
    out_analytical = []
    out_q = []

    # Inventory at the start of each step — needed for the HMM likelihood,
    # which uses the inventory *before* the fills resolved.
    def get_q(state):
        # State may be wrapped; q lives at state["q"] for unwrapped MM, or
        # state["inner"]["q"] / state["q"] for wrappers (BeliefObsEnv keeps
        # MM keys at top level). Try both.
        if "q" in state:
            return state["q"]
        return state["inner"]["q"]

    # Per-step act (vmapped over rollouts)
    @jax.jit
    def act_step(agent_state_, carry_, obs_, key_):
        keys = jax.random.split(key_, n_rollouts)
        return jax.vmap(
            lambda c, o, k: agent.act(agent_state_, c, o, k),
            in_axes=(0, 0, 0),
        )(carry_, obs_, keys)

    @jax.jit
    def hmm_step_v(b, action, bid_fill, ask_fill, q, done):
        # full_update returns (b_filtered_t, b_pred_{t+1}). The wrapper
        # uses b_pred for the next step's prior; on done, reset to initial.
        def per_env(b_i, a_i, bf_i, af_i, q_i, d_i):
            b_filt, b_pred = full_update(b_i, inner_mm, a_i, bf_i, af_i, q_i)
            b_init = initial_belief(inner_mm)
            b_next = jnp.where(d_i, b_init, b_pred)
            # b_filt is the posterior at t (what we want to compare to method's
            # belief if the method's belief is "after-fills"). For consistency
            # with how each method updates: VariBAD μ is updated *during* the
            # GRU forward pass with current observation, before action; RL²'s
            # carry similarly. So b_filt at time t represents posterior given
            # all evidence through step t. We record b_filt as the analytical
            # belief at time t (the value the method's belief is implicitly
            # approximating).
            return b_filt, b_next
        return jax.vmap(per_env)(b, action, bid_fill, ask_fill, q, done)

    for t in range(rollout_length):
        # Belief BEFORE this step (the posterior the agent uses for action_t)
        # For RL²: carry going into act. For VariBAD: μ produced by encoder
        # given current obs. The latter is in act() extras.
        carry_pre_act = carry  # for RL²

        # Inventory at start of step (for HMM likelihood)
        q_pre = jax.vmap(get_q)(env_states)

        step_key, key = jax.random.split(key)
        action, extras, new_carry = act_step(agent_state, carry, obses, step_key)

        # Method-specific belief at time t. For RL² belief_key="carry_in"
        # which is the carry going INTO act → carry_pre_act. For VariBAD
        # belief_key="mu" which is in extras.
        if belief_key == "carry_in":
            belief_t = carry_pre_act
        else:
            if belief_key not in extras:
                raise KeyError(
                    f"belief_key {belief_key!r} not in agent.act extras "
                    f"(have: {list(extras.keys())})"
                )
            belief_t = extras[belief_key]

        # Step env
        step_keys = jax.random.split(key, n_rollouts + 1)
        env_keys, key = step_keys[:n_rollouts], step_keys[-1]
        env_states, obses_next, rewards, dones, info = step_v(
            env_states, action, env_keys
        )

        # Analytical belief update at time t (uses bid_fill, ask_fill from
        # info, plus q_pre, plus action). The b_filt return is the posterior
        # at time t.
        b_filt_t, analytical_belief = hmm_step_v(
            analytical_belief,
            action,
            info["bid_fill"],
            info["ask_fill"],
            q_pre,
            dones,
        )

        out_obs.append(np.asarray(obses))
        out_belief.append(np.asarray(belief_t))
        out_regime.append(np.asarray(info["regime"]))
        out_action.append(np.asarray(action))
        out_reward.append(np.asarray(rewards))
        out_done.append(np.asarray(dones))
        out_analytical.append(np.asarray(b_filt_t))
        out_q.append(np.asarray(q_pre))

        # Reset carry on episode boundary (matches training behavior)
        zeros = jnp.zeros_like(new_carry)
        mask = dones[:, None].astype(new_carry.dtype)
        carry = mask * zeros + (1.0 - mask) * new_carry
        obses = obses_next

    return {
        "obs": np.stack(out_obs),  # [T, N, obs_dim]
        "belief": np.stack(out_belief),  # [T, N, belief_dim]
        "regime": np.stack(out_regime).astype(np.int32),  # [T, N]
        "action": np.stack(out_action).astype(np.int32),  # [T, N]
        "reward": np.stack(out_reward),  # [T, N]
        "done": np.stack(out_done).astype(np.int32),  # [T, N]
        "analytical_belief": np.stack(out_analytical),  # [T, N, n_regimes]
        "q": np.stack(out_q),  # [T, N]
    }


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------


def train_probe(
    belief_TND: np.ndarray,  # [T, N, D]
    regime_TN: np.ndarray,  # [T, N]
    train_rollout_idx: np.ndarray,  # [N_train]
    test_rollout_idx: np.ndarray,  # [N_test]
    classifier: str = "logistic",
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a regime classifier from belief→regime, return train/test accuracies.

    Uses *all* timesteps from training rollouts as flat dataset, but a
    rollout-level train/test split (so test rollouts are unseen).

    Returns dict with:
      - train_acc, test_acc: scalar accuracies (over all timesteps in split)
      - per_t_test_acc: [T] per-timestep accuracy on test rollouts
      - n_classes: number of regime classes seen in training
    """
    T, N, D = belief_TND.shape

    train_belief = belief_TND[:, train_rollout_idx, :].reshape(-1, D)
    train_regime = regime_TN[:, train_rollout_idx].reshape(-1)
    test_belief = belief_TND[:, test_rollout_idx, :].reshape(-1, D)
    test_regime = regime_TN[:, test_rollout_idx].reshape(-1)

    # StandardScaler avoids overflow when raw GRU hidden states have large
    # magnitudes; C=0.1 keeps L-BFGS weights bounded, which prevents
    # overflow on degenerate inputs like the 3-simplex analytical posterior
    # (sum-to-1 creates a flat direction in the multinomial logistic loss).
    if classifier == "logistic":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs", max_iter=500, C=0.1, random_state=seed,
            ),
        )
    elif classifier == "mlp":
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,), max_iter=200, alpha=0.01,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"unknown classifier: {classifier!r}")

    with _suppress_blas_matmul_warnings():
        clf.fit(train_belief, train_regime)
        train_acc = float(clf.score(train_belief, train_regime))
        test_acc = float(clf.score(test_belief, test_regime))

        # Per-timestep accuracy on test rollouts.
        test_belief_TND = belief_TND[:, test_rollout_idx, :]  # [T, N_test, D]
        test_regime_TN = regime_TN[:, test_rollout_idx]  # [T, N_test]
        per_t = np.zeros(T, dtype=float)
        for t in range(T):
            preds = clf.predict(test_belief_TND[t])
            per_t[t] = float((preds == test_regime_TN[t]).mean())

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "per_t_test_acc": per_t.tolist(),
        "n_classes": int(clf.classes_.size),
    }


# ---------------------------------------------------------------------------
# Experiment loading (config + checkpoint → env, agent, agent_state)
# ---------------------------------------------------------------------------


@dataclass
class ProbeBundle:
    env: Any
    agent: Any
    agent_state: chex.ArrayTree
    config: dict[str, Any]
    seed: int


def load_experiment(exp_dir: Path | str, seed: int) -> ProbeBundle:
    """Load env, agent, and trained agent_state for a given experiment+seed.

    Reuses `training.train._build_env`, `_build_agent`, `_maybe_wrap_env_for_agent`
    so the env/agent are reconstructed exactly as during training.
    """
    from training.config import ExperimentConfig, EnvConfig, AgentConfig
    from training.train import _build_env, _build_agent, _maybe_wrap_env_for_agent

    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json") as f:
        cfg_raw = json.load(f)

    # Reconstruct ExperimentConfig from the saved dict.
    cfg = ExperimentConfig(
        experiment_name=cfg_raw["experiment_name"],
        env=EnvConfig(name=cfg_raw["env"]["name"], params=dict(cfg_raw["env"]["params"])),
        agent=AgentConfig(name=cfg_raw["agent"]["name"], params=dict(cfg_raw["agent"]["params"])),
        iterations=int(cfg_raw["iterations"]),
        parallel_envs=int(cfg_raw["parallel_envs"]),
        rollout_length=int(cfg_raw["rollout_length"]),
        num_seeds=int(cfg_raw["num_seeds"]),
        seed_base=int(cfg_raw["seed_base"]),
        run_mode=cfg_raw.get("run_mode", "full"),
    )

    env = _build_env(cfg)
    env = _maybe_wrap_env_for_agent(cfg, env)
    agent = _build_agent(cfg, env)

    ckpt_path = exp_dir / f"checkpoint_seed_{seed}.pkl"
    with open(ckpt_path, "rb") as f:
        agent_state = pickle.load(f)

    return ProbeBundle(
        env=env, agent=agent, agent_state=agent_state, config=cfg_raw, seed=seed
    )


# ---------------------------------------------------------------------------
# End-to-end probe for one method × one seed
# ---------------------------------------------------------------------------


def probe_one_seed(
    bundle: ProbeBundle,
    n_rollouts: int = 200,
    rollout_length: int = 128,
    classifier: str = "logistic",
    test_frac: float = 0.2,
    rng_key: int = 0,
) -> dict[str, Any]:
    """Full probe pipeline for one (method, seed) checkpoint.

    Returns a dict with method-belief and analytical-belief probe results.
    """
    key = jax.random.PRNGKey(rng_key)
    data = collect_probe_rollouts(
        bundle.env, bundle.agent, bundle.agent_state,
        n_rollouts=n_rollouts, rollout_length=rollout_length, key=key,
    )

    # Train/test split by rollout index (not timestep).
    rng = np.random.default_rng(rng_key)
    perm = rng.permutation(n_rollouts)
    n_test = max(1, int(round(n_rollouts * test_frac)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    method_probe = train_probe(
        data["belief"], data["regime"], train_idx, test_idx,
        classifier=classifier, seed=rng_key,
    )
    analytical_probe = train_probe(
        data["analytical_belief"], data["regime"], train_idx, test_idx,
        classifier=classifier, seed=rng_key,
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
