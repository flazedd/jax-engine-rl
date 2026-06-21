"""Frozen-encoder PPO transplant (RL², medium env).

Loads a trained RL²-concat encoder (Dense_0 + GRUCell_0), FREEZES it, attaches a
freshly-initialised policy+value head (concat or hypernet), and trains ONLY the
head with PPO. Tests whether the deployed RL²-concat failure (return ~108, while
its belief supports ~168 by BC) is a JOINT-optimisation problem: if PPO on the
frozen good belief now reaches ~150-168, decoupling the encoder fixes it; if it
stalls near ~108, concat has an intrinsic RL pathology.

Freezing: subclass RL2Agent and override `_optimizer()` to use
optax.multi_transform with set_to_zero on the encoder (top-level params/Dense_0
and params/GRUCell_0), training everything else.

Usage:
  uv run python -m scripts.m5r_frozen_encoder_transplant --head concat --seeds 0 1 2 --iterations 200
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.rl2 import RL2Agent
from training.train import _make_iter_step
from evaluation.posterior_probe import load_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
ENC_EXP = "m5r_final_rl2_concat_e_final"
FINAL = RESULTS_ROOT / "M5R" / "final"
FROZEN = ("Dense_0", "GRUCell_0")  # top-level encoder modules


class FrozenEncoderRL2(RL2Agent):
    """RL² whose GRU encoder (Dense_0 + GRUCell_0) is frozen during PPO."""

    def _optimizer(self) -> optax.GradientTransformation:
        def label(path, _x):
            ks = [str(k.key) for k in path]
            frozen = len(ks) >= 2 and ks[0] == "params" and ks[1] in FROZEN
            return "frozen" if frozen else "train"

        return optax.multi_transform(
            {
                "train": optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adam(self.learning_rate),
                ),
                "frozen": optax.set_to_zero(),
            },
            lambda params: jax.tree_util.tree_map_with_path(label, params),
        )


def _as_frozen_agent(agent: RL2Agent) -> FrozenEncoderRL2:
    return FrozenEncoderRL2(**{f.name: getattr(agent, f.name) for f in dataclasses.fields(agent)})


def _enc_l2(params):
    return {k: float(jnp.sqrt(sum(jnp.sum(x ** 2)
            for x in jax.tree_util.tree_leaves(params["params"][k])))) for k in FROZEN}


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.m5r_frozen_encoder_transplant")
    ap.add_argument("--head", choices=["concat", "hypernet"], default="concat")
    ap.add_argument("--enc-exp", default=ENC_EXP,
                    help="encoder experiment dir under results/ to freeze and transplant onto")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--parallel-envs", type=int, default=512)
    ap.add_argument("--rollout-length", type=int, default=128)
    args = ap.parse_args()

    enc_dir = RESULTS_ROOT / args.enc_exp
    enc_tag = args.enc_exp.replace("m5r_final_rl2_concat_", "")
    returns_final = []
    per_seed = {}

    for seed in args.seeds:
        bundle = load_experiment(enc_dir, seed)
        base = bundle.agent  # RL2Agent concat, correct obs_size/n_actions/hidden_dim
        if args.head == "hypernet":
            base = dataclasses.replace(
                base, integration="hypernet",
                hypernet_hidden=16, hypernet_target_hidden=16, hypernet_init_scale=0.01,
            )
        else:
            base = dataclasses.replace(base, integration="concat")
        agent = _as_frozen_agent(base)
        env = bundle.env

        # fresh init, then graft the trained encoder
        state = agent.init(jax.random.PRNGKey(1000 + seed))
        enc_src = bundle.agent_state["params"]["params"]
        for k in FROZEN:
            state["params"]["params"][k] = enc_src[k]
        # re-init opt_state for the (masked) optimizer on grafted params
        state["opt_state"] = agent._optimizer().init(state["params"])
        enc_before = _enc_l2(state["params"])

        step_fn = _make_iter_step(env, agent, args.parallel_envs, args.rollout_length)
        key = jax.random.PRNGKey(seed)
        curve = []
        for it in range(args.iterations):
            state, key, metrics = step_fn(state, key)
            r = float(metrics["mean_return"])
            curve.append(r)
            if it == 0 or (it + 1) % max(1, args.iterations // 8) == 0:
                print(f"[transplant {args.head}] seed {seed} iter {it+1}/{args.iterations} return={r:.1f}", flush=True)

        enc_after = _enc_l2(state["params"])
        # head changed? compare a head leaf norm
        final = float(np.mean(curve[-50:])) if len(curve) >= 50 else float(np.mean(curve))
        returns_final.append(final)
        per_seed[seed] = {
            "final_return": final,
            "curve_tail": curve[-5:],
            "enc_l2_before": enc_before,
            "enc_l2_after": enc_after,
        }
        drift = {k: abs(enc_after[k] - enc_before[k]) for k in FROZEN}
        print(f"[transplant {args.head}] seed {seed} FINAL return={final:.1f} | "
              f"encoder drift (should be ~0): {drift}", flush=True)

    summary = {
        "head": args.head, "enc_exp": args.enc_exp, "seeds": args.seeds, "iterations": args.iterations,
        "final_return_mean": float(np.mean(returns_final)),
        "final_return_std": float(np.std(returns_final)),
        "per_seed": per_seed,
        "references": {"deployed_concat": 108.6, "deployed_hypernet": 158.9,
                       "floor": 137.8, "belief_ceiling": 169.9, "bc_ceiling_on_belief": 168.0},
    }
    FINAL.mkdir(parents=True, exist_ok=True)
    out_path = FINAL / f"m5r_transplant_{enc_tag}_{args.head}.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\n[transplant {args.head}] === mean final return = "
          f"{summary['final_return_mean']:.1f} ± {summary['final_return_std']:.1f} "
          f"(refs: deployed-concat 108.6 | floor 137.8 | BC-ceiling 168 | deployed-hypernet 158.9) ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
