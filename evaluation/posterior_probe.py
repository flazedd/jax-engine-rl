"""Posterior-quality probe for meta-RL methods on MM E_final.

Given a trained checkpoint, run evaluation rollouts on the env, record
the agent's internal belief representation (GRU hidden for RL², posterior
μ for VariBAD), and the env's true regime label per timestep. Train a
classifier from belief → regime; report mean test accuracy and per-
timestep accuracy curves.

Spec lives at `results/milestones/M5/STEP5_PROBE_DESIGN.md`.

Currently MM-specific: assumes the env is `MarketMakingV1` (possibly
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
from sklearn.metrics import log_loss as _sk_log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _brier(regime: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Mean multiclass Brier score: E[ sum_c (q_c - 1{z=c})^2 ]. Range [0, 2]."""
    onehot = np.eye(n_classes)[regime]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def _kl_omega_q(omega: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Mean forward KL(omega || q) per sample.

    omega, q: [M, C] rows are distributions over the C regimes. omega is the
    analytical (Bayes-optimal) belief, q the probe's recovered belief. Forward
    KL is finite here because q comes from a softmax (strictly > 0); where
    omega_c = 0 the term is taken as 0 (0 * log 0 = 0).
    """
    q = np.clip(q, eps, 1.0)
    omega = np.clip(omega, 0.0, 1.0)
    omega = omega / np.clip(omega.sum(axis=1, keepdims=True), eps, None)
    log_omega = np.log(np.clip(omega, eps, 1.0))
    term = np.where(omega > 0.0, omega * (log_omega - np.log(q)), 0.0)
    return float(term.sum(axis=1).mean())


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
from envs.market_making_v1 import MarketMakingV1


# ---------------------------------------------------------------------------
# Inner-env extraction (walk through wrappers to find MarketMakingV1)
# ---------------------------------------------------------------------------


def get_inner_mm(env: Any) -> MarketMakingV1:
    """Walk down wrapper chain until we find the MarketMakingV1."""
    cur = env
    while not isinstance(cur, MarketMakingV1):
        if not hasattr(cur, "inner"):
            raise TypeError(
                f"could not find MarketMakingV1 inside {type(cur).__name__} "
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
    extra_keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Run `n_rollouts` parallel evaluation rollouts.

    `extra_keys` names additional entries of the agent's `act` extras to record
    alongside the belief, each stacked to `[T, N, ...]` under the same name.
    The belief-swap diagnostic uses it to capture VariBAD's `log_var`, which
    travels with `mu` into the policy and cannot be substituted without it.

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
    out_extra: dict[str, list] = {k: [] for k in extra_keys}

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

        for k in extra_keys:
            if k not in extras:
                raise KeyError(
                    f"extra key {k!r} not in agent.act extras "
                    f"(have: {list(extras.keys())})"
                )
            out_extra[k].append(np.asarray(extras[k]))

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

    out = {
        "obs": np.stack(out_obs),  # [T, N, obs_dim]
        "belief": np.stack(out_belief),  # [T, N, belief_dim]
        "regime": np.stack(out_regime).astype(np.int32),  # [T, N]
        "action": np.stack(out_action).astype(np.int32),  # [T, N]
        "reward": np.stack(out_reward),  # [T, N]
        "done": np.stack(out_done).astype(np.int32),  # [T, N]
        "analytical_belief": np.stack(out_analytical),  # [T, N, n_regimes]
        "q": np.stack(out_q),  # [T, N]
    }
    for k in extra_keys:
        out[k] = np.stack(out_extra[k])  # [T, N, ...]
    return out


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
    omega_TND: np.ndarray | None = None,  # [T, N, C] analytical belief, for KL
) -> dict[str, Any]:
    """Fit a regime classifier from belief→regime, return belief-quality metrics.

    Uses *all* timesteps from training rollouts as flat dataset, but a
    rollout-level train/test split (so test rollouts are unseen). The fitted
    probe's softmax over the test set is scored as a *distribution*, not just
    by its argmax, so overconfident-but-wrong beliefs are penalised.

    Returns dict with:
      - train_acc, test_acc: scalar argmax accuracies (mode-match rate)
      - test_log_loss: mean cross-entropy of the probe softmax vs the true
        regime label (proper scoring rule; lower is better)
      - test_brier: mean multiclass Brier score (proper, bounded [0, 2])
      - test_kl_to_omega (only if `omega_TND` given): mean forward
        KL(omega || q) of the analytical belief from the probe softmax, i.e.
        distance of the recovered belief from the Bayes-optimal ceiling belief
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

        # Probe softmax over the test set, scattered into full regime columns
        # (a class absent from training gets probability 0).
        classes = clf.classes_.astype(int)
        n_cols = int(max(int(regime_TN.max()) + 1, int(classes.max()) + 1))
        raw_proba = clf.predict_proba(test_belief)  # [M, len(classes)]
        test_proba = np.zeros((raw_proba.shape[0], n_cols), dtype=float)
        test_proba[:, classes] = raw_proba
        labels = list(range(n_cols))
        test_log_loss = float(_sk_log_loss(test_regime, test_proba, labels=labels))
        test_brier = _brier(test_regime, test_proba, n_cols)

        kl_to_omega: float | None = None
        if omega_TND is not None:
            omega_test = omega_TND[:, test_rollout_idx, :].reshape(
                -1, omega_TND.shape[-1]
            )
            if omega_test.shape[1] < n_cols:
                pad = np.zeros((omega_test.shape[0], n_cols - omega_test.shape[1]))
                omega_test = np.concatenate([omega_test, pad], axis=1)
            kl_to_omega = _kl_omega_q(omega_test, test_proba)

        # Per-timestep accuracy on test rollouts.
        test_belief_TND = belief_TND[:, test_rollout_idx, :]  # [T, N_test, D]
        test_regime_TN = regime_TN[:, test_rollout_idx]  # [T, N_test]
        per_t = np.zeros(T, dtype=float)
        for t in range(T):
            preds = clf.predict(test_belief_TND[t])
            per_t[t] = float((preds == test_regime_TN[t]).mean())

    out = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "test_log_loss": test_log_loss,
        "test_brier": test_brier,
        "per_t_test_acc": per_t.tolist(),
        "n_classes": int(clf.classes_.size),
    }
    if kl_to_omega is not None:
        out["test_kl_to_omega"] = kl_to_omega
    return out


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


def load_experiment(
    exp_dir: Path | str,
    seed: int,
    env_overrides: dict | None = None,
) -> ProbeBundle:
    """Load env, agent, and trained agent_state for a given experiment+seed.

    Reuses `training.train._build_env`, `_build_agent`, `_maybe_wrap_env_for_agent`
    so the env/agent are reconstructed exactly as during training.

    `env_overrides` patches the saved env params before the env is built. The
    agent is unaffected, so the trained policy is evaluated unchanged on a
    modified environment. Used to set `lock_regime` for the action-distribution
    diagnostic, which needs the regime held fixed rather than switching.
    """
    from training.config import ExperimentConfig, EnvConfig, AgentConfig
    from training.train import _build_env, _build_agent, _maybe_wrap_env_for_agent

    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json") as f:
        cfg_raw = json.load(f)

    env_params = dict(cfg_raw["env"]["params"])
    if env_overrides:
        env_params.update(env_overrides)

    # Reconstruct ExperimentConfig from the saved dict.
    cfg = ExperimentConfig(
        experiment_name=cfg_raw["experiment_name"],
        env=EnvConfig(name=cfg_raw["env"]["name"], params=env_params),
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

    # The analytical belief (HMM posterior) is the reference the method's
    # recovered belief is scored against via KL. It is passed as omega to the
    # method probe; for the analytical probe it doubles as a self-consistency
    # check (KL of omega from a probe re-fit on omega, expected near zero).
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
