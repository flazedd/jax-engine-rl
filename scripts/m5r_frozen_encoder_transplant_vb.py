"""Frozen-encoder PPO transplant for VariBAD (medium env) — generality check.

Mirrors scripts/m5r_frozen_encoder_transplant.py but for VariBAD. Loads a trained
VariBAD-concat checkpoint, FREEZES its variational encoder and reward decoder
(top-level "encoder" and "decoder" param collections), attaches a freshly
initialised policy head (concat or hypernet), and trains ONLY the policy with PPO.
If the RL² result generalises, concat should stall near its deployed return while
hypernet reaches the belief ceiling on the same frozen belief.

Usage:
  uv run python -m scripts.m5r_frozen_encoder_transplant_vb --head hypernet --seeds 0 1 2 --iterations 200
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

from agents.varibad import VariBADAgent
from training.train import _make_iter_step
from evaluation.posterior_probe import load_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
ENC_EXP = "m5r_final_varibad_concat_e_final"
FINAL = RESULTS_ROOT / "M5R" / "final"
FROZEN = ("encoder", "decoder")  # top-level collections to freeze


class FrozenEncoderVariBAD(VariBADAgent):
    """VariBAD whose encoder + decoder are frozen during PPO (policy only)."""

    def _optimizer(self) -> optax.GradientTransformation:
        def label(path, _x):
            ks = [str(k.key) for k in path]
            return "frozen" if (ks and ks[0] in FROZEN) else "train"

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


def _as_frozen(agent: VariBADAgent) -> FrozenEncoderVariBAD:
    return FrozenEncoderVariBAD(**{f.name: getattr(agent, f.name) for f in dataclasses.fields(agent)})


def _enc_l2(params):
    return float(jnp.sqrt(sum(jnp.sum(x ** 2)
            for x in jax.tree_util.tree_leaves(params["encoder"]))))


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.m5r_frozen_encoder_transplant_vb")
    ap.add_argument("--head", choices=["concat", "hypernet"], default="concat")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--parallel-envs", type=int, default=512)
    ap.add_argument("--rollout-length", type=int, default=128)
    args = ap.parse_args()

    enc_dir = RESULTS_ROOT / ENC_EXP
    finals, per_seed = [], {}
    for seed in args.seeds:
        bundle = load_experiment(enc_dir, seed)
        base = bundle.agent
        if args.head == "hypernet":
            base = dataclasses.replace(
                base, integration="hypernet",
                hypernet_hidden=16, hypernet_target_hidden=16, hypernet_init_scale=0.01)
        else:
            base = dataclasses.replace(base, integration="concat")
        agent = _as_frozen(base)
        env = bundle.env

        state = agent.init(jax.random.PRNGKey(1000 + seed))
        for k in FROZEN:
            state["params"][k] = bundle.agent_state["params"][k]
        state["opt_state"] = agent._optimizer().init(state["params"])
        enc_before = _enc_l2(state["params"])

        step_fn = _make_iter_step(env, agent, args.parallel_envs, args.rollout_length)
        key = jax.random.PRNGKey(seed)
        curve = []
        for it in range(args.iterations):
            state, key, metrics = step_fn(state, key)
            curve.append(float(metrics["mean_return"]))
            if it == 0 or (it + 1) % max(1, args.iterations // 8) == 0:
                print(f"[vb-transplant {args.head}] seed {seed} iter {it+1}/{args.iterations} "
                      f"return={curve[-1]:.1f}", flush=True)
        enc_after = _enc_l2(state["params"])
        final = float(np.mean(curve[-50:])) if len(curve) >= 50 else float(np.mean(curve))
        finals.append(final)
        per_seed[seed] = {"final_return": final, "enc_drift": abs(enc_after - enc_before)}
        print(f"[vb-transplant {args.head}] seed {seed} FINAL return={final:.1f} | "
              f"encoder drift (should be ~0): {abs(enc_after - enc_before):.1e}", flush=True)

    summary = {
        "head": args.head, "enc_exp": ENC_EXP, "seeds": args.seeds,
        "iterations": args.iterations,
        "final_return_mean": float(np.mean(finals)),
        "final_return_std": float(np.std(finals)),
        "per_seed": per_seed,
        "references": {"deployed_vb_concat": 103.8, "deployed_vb_hypernet": 160.2,
                       "floor": 137.8, "belief_ceiling": 169.9},
    }
    FINAL.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(FINAL / f"m5r_transplant_vb_{args.head}.json", "w"), indent=2)
    print(f"\n[vb-transplant {args.head}] === mean final return = "
          f"{summary['final_return_mean']:.1f} ± {summary['final_return_std']:.1f} "
          f"(refs: deployed-VB-concat 103.8 | floor 137.8 | deployed-VB-hypernet 160.2 | belief 169.9) ===",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
