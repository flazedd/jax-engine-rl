"""Closed-loop RETURN of BC-trained policies on frozen meta-RL beliefs (RL²).

Does the high BC action-match (~0.91) on RL²-concat's belief translate into
good RETURN when deployed closed-loop? We freeze the RL² encoder, train a head
(concat or hypernet) to imitate the VI-optimal action from the belief, then roll
out (frozen encoder + BC head) in the env and measure episode return. If acting
VI-optimally on RL²-concat's belief yields a high return (near the belief
ceiling), the belief is good for control and the original concat RL failure was
an optimization problem, not a representation one.

The RL² obs wrapper feeds the previous action back into the augmented obs, so
the recurrent belief propagates correctly under the BC actions (closed loop).

Sanity: rolling out the original policy (its own sampled actions) should match
the method's deployed return (RL² concat ≈ 108, RL² hypernet ≈ 158).

Output: results/M5R/final/m5r_bc_rollout_return.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from envs.market_making_v1 import MarketMakingV1
from oracles.value_iteration import solve_value_iteration
from evaluation.posterior_probe import collect_probe_rollouts, load_experiment
from scripts.m5r_bc_belief_usage import MLPHead, HypernetHead

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FINAL = RESULTS_ROOT / "M5R" / "final"
ENV = "e_final"
N_ACTIONS = 3
TEST_FRAC = 0.2


def _get_q(state):
    return state["q"] if "q" in state else state["inner"]["q"]


def _train_head(model, obs, bel, y, key, steps, lr=1e-2):
    params = model.init(key, obs[:2], bel[:2])

    def loss_fn(p, ob, be, yy):
        return optax.softmax_cross_entropy_with_integer_labels(
            model.apply(p, ob, be), yy).mean()

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, os_, ob, be, yy):
        loss, g = jax.value_and_grad(loss_fn)(p, ob, be, yy)
        upd, os_ = opt.update(g, os_, p)
        return optax.apply_updates(p, upd), os_, loss

    for _ in range(steps):
        params, opt_state, _ = step(params, opt_state, obs, bel, y)
    return params


def _rollout_return(agent, agent_state, env, head_model, head_params, belief_key,
                    n_rollouts, H, key):
    """Closed-loop episode return [n_rollouts]. If head_params is None, use the
    agent's own sampled action (sanity); else use argmax of the BC head."""
    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)
    keys = jax.random.split(key, n_rollouts + 1)
    env_states, obs = reset_v(keys[:n_rollouts])
    key = keys[-1]
    carry = agent.init_carry(n_rollouts)
    total = jnp.zeros(n_rollouts)

    head_apply = None
    if head_params is not None:
        head_apply = jax.jit(lambda ob, be: head_model.apply(head_params, ob, be))

    for _ in range(H):
        k, key = jax.random.split(key)
        aks = jax.random.split(k, n_rollouts)
        action_o, extras, new_carry = jax.vmap(
            lambda c, o, kk: agent.act(agent_state, c, o, kk)
        )(carry, obs, aks)
        if head_apply is None:
            action = action_o
        else:
            belief = carry if belief_key == "carry_in" else extras[belief_key]
            action = jnp.argmax(head_apply(obs, belief), axis=-1)
        sks = jax.random.split(key, n_rollouts + 1)
        env_keys, key = sks[:n_rollouts], sks[-1]
        env_states, obs, rewards, dones, info = step_v(env_states, action, env_keys)
        total = total + rewards
        carry = jnp.where(dones[:, None], jnp.zeros_like(new_carry), new_carry)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.m5r_bc_rollout_return")
    ap.add_argument("--methods", nargs="+", default=["rl2_concat", "rl2_hypernet"])
    ap.add_argument("--n-rollouts", type=int, default=400)
    ap.add_argument("--rollout-length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    stats = json.load(open(RESULTS_ROOT / "milestones" / "M2" / "stats_M2_requirements.json"))
    mm = MarketMakingV1(**stats["parameters"])
    vi_policy = np.asarray(solve_value_iteration(mm).policy)
    inv_max = int(mm.inventory_max)

    out: dict[str, dict[str, list[float]]] = {}
    for method in args.methods:
        exp_dir = RESULTS_ROOT / f"m5r_final_{method}_{ENV}"
        seeds = sorted(int(p.stem.split("_")[-1]) for p in exp_dir.glob("checkpoint_seed_*.pkl"))
        for seed in seeds:
            bundle = load_experiment(exp_dir, seed)
            agent, agent_state, env = bundle.agent, bundle.agent_state, bundle.env
            belief_key = getattr(agent, "belief_key")

            # 1) collect rollouts under the original policy to train BC heads
            data = collect_probe_rollouts(
                env, agent, agent_state, n_rollouts=args.n_rollouts,
                rollout_length=args.rollout_length, key=jax.random.PRNGKey(seed),
            )
            T, N, _ = data["belief"].shape
            obs = jnp.asarray(data["obs"].reshape(T * N, -1).astype(np.float32))
            bel = jnp.asarray(data["belief"].reshape(T * N, -1).astype(np.float32))
            reg = data["regime"].reshape(T * N)
            q = data["q"].reshape(T * N)
            inv_idx = np.clip(np.rint(q).astype(int) + inv_max, 0, 2 * inv_max)
            y = jnp.asarray(vi_policy[inv_idx, reg].astype(np.int32))
            obs_dim = obs.shape[1]

            concat = MLPHead(hidden=(48,), n_actions=N_ACTIONS)
            hyper = HypernetHead(obs_dim=obs_dim, target_hidden=16,
                                 n_actions=N_ACTIONS, hypernet_hidden=16)
            p_concat = _train_head(concat, obs, bel, y, jax.random.PRNGKey(seed), args.steps)
            p_hyper = _train_head(hyper, obs, bel, y, jax.random.PRNGKey(seed + 1), args.steps)

            # 2) closed-loop returns
            rk = jax.random.PRNGKey(seed + 100)
            ret_orig = _rollout_return(agent, agent_state, env, None, None, belief_key,
                                       args.n_rollouts, args.rollout_length, rk)
            ret_concat = _rollout_return(agent, agent_state, env, concat, p_concat, belief_key,
                                         args.n_rollouts, args.rollout_length, rk)
            ret_hyper = _rollout_return(agent, agent_state, env, hyper, p_hyper, belief_key,
                                        args.n_rollouts, args.rollout_length, rk)

            out.setdefault(method, {}).setdefault("orig", []).append(float(ret_orig.mean()))
            out[method].setdefault("bc_concat_head", []).append(float(ret_concat.mean()))
            out[method].setdefault("bc_hypernet_head", []).append(float(ret_hyper.mean()))
            print(f"[bcret] {method} seed {seed}: orig={float(ret_orig.mean()):.1f} "
                  f"bc_concat={float(ret_concat.mean()):.1f} "
                  f"bc_hyper={float(ret_hyper.mean()):.1f}", flush=True)

    def summ(xs):
        a = np.asarray(xs)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    summary = {m: {k: summ(v) for k, v in d.items()} for m, d in out.items()}
    FINAL.mkdir(parents=True, exist_ok=True)
    json.dump({"n_rollouts": args.n_rollouts, "summary": summary, "raw": out},
              open(FINAL / "m5r_bc_rollout_return.json", "w"), indent=2)
    print("\n[bcret] === summary (mean episode return across seeds) ===", flush=True)
    print("references: floor≈137.8  RL²-concat-deployed≈107.8  RL²-hypernet-deployed≈158.4 "
          " belief≈169.9  oracle≈181.6", flush=True)
    for m, d in summary.items():
        print(f"  {m:>14}: orig={d['orig']['mean']:.1f}  "
              f"bc_concat_head={d['bc_concat_head']['mean']:.1f}  "
              f"bc_hypernet_head={d['bc_hypernet_head']['mean']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
