"""Single-stage 'best of both': RL² hypernet + auxiliary regime-decoding loss.

End-to-end training (encoder NOT frozen) of RL² with a hypernet policy head and
an auxiliary linear regime-decoder on the GRU belief, weighted by
`--aux-coef`. Goal: get concat-quality (decodable) belief AND hypernet usage in
a single run, beating both deployed pure methods.

Reports, per aux coefficient:
  - final return (mean over final 50 iters), vs deployed hypernet 158.9
  - belief decodability: post-hoc logistic probe accuracy on the trained belief
    (deployed concat 0.70, deployed hypernet 0.56)

aux-coef = 0 reproduces plain RL² hypernet (control).

Usage:
  uv run python -m scripts.m5r_single_stage_auxdecode --aux-coefs 0 0.3 1.0 --seeds 0 1 2 --iterations 200
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import numpy as np

from training.train import _make_iter_step
from evaluation.posterior_probe import (
    collect_probe_rollouts, load_experiment, train_probe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FINAL = RESULTS_ROOT / "M5R" / "final"
# Source of env + config + base (concat) agent, per method.
ENC_EXP = {
    "rl2": "m5r_final_rl2_concat_e_final",
    "varibad": "m5r_final_varibad_concat_e_final",
}
# Reference levels for the summary print, per method (deployed pure cells +
# Belief-PPO ceiling), from the medium-environment results.
REFS = {
    "rl2": "deployed-concat 118.7 | deployed-hypernet 163.1 | belief 168.8",
    "varibad": "deployed-concat 108.9 | deployed-hypernet 163.5 | belief 168.8",
}


def _probe_acc(agent, agent_state, env, seed, n_rollouts=400):
    data = collect_probe_rollouts(env, agent, agent_state,
                                  n_rollouts=n_rollouts, rollout_length=128,
                                  key=jax.random.PRNGKey(seed + 7))
    N = data["belief"].shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_test = max(1, int(round(N * 0.2)))
    res = train_probe(data["belief"], data["regime"], perm[n_test:], perm[:n_test],
                      classifier="logistic", seed=seed)
    return float(res["test_acc"])


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.m5r_single_stage_auxdecode")
    ap.add_argument("--method", default="rl2", choices=["rl2", "varibad"])
    ap.add_argument("--aux-coefs", type=float, nargs="+", default=[0.0, 0.3, 1.0])
    ap.add_argument("--detach", action="store_true",
                    help="detach belief into policy/value heads so the encoder is "
                         "shaped only by the auxiliary regime-decode loss")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--parallel-envs", type=int, default=512)
    ap.add_argument("--rollout-length", type=int, default=128)
    args = ap.parse_args()

    bundle = load_experiment(RESULTS_ROOT / ENC_EXP[args.method], 0)
    env = bundle.env
    base = bundle.agent  # concat agent with correct obs_size/n_actions/hidden_dim

    out = {}
    for coef in args.aux_coefs:
        rets, accs = [], []
        for seed in args.seeds:
            agent = dataclasses.replace(
                base, integration="hypernet",
                hypernet_hidden=16, hypernet_target_hidden=16, hypernet_init_scale=0.01,
                aux_decode_coef=coef, n_regimes=int(base.n_regimes),
                detach_belief_for_policy=args.detach,
            )
            state = agent.init(jax.random.PRNGKey(1000 + seed))
            step_fn = _make_iter_step(env, agent, args.parallel_envs, args.rollout_length)
            key = jax.random.PRNGKey(seed)
            curve, aux_acc = [], None
            for it in range(args.iterations):
                state, key, metrics = step_fn(state, key)
                curve.append(float(metrics["mean_return"]))
                if "aux/decode_acc" in metrics:
                    aux_acc = float(metrics["aux/decode_acc"])
            final_ret = float(np.mean(curve[-50:])) if len(curve) >= 50 else float(np.mean(curve))
            probe = _probe_acc(agent, state, env, seed)
            rets.append(final_ret); accs.append(probe)
            print(f"[single-stage] coef={coef} seed={seed}: return={final_ret:.1f} "
                  f"probe_acc={probe:.3f}" + (f" aux_acc={aux_acc:.3f}" if aux_acc is not None else ""),
                  flush=True)
        out[str(coef)] = {
            "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "probe_acc_mean": float(np.mean(accs)), "probe_acc_std": float(np.std(accs)),
            "returns": rets, "probe_accs": accs,
        }

    FINAL.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    json.dump({"iterations": args.iterations, "seeds": args.seeds, "detach": args.detach,
               "by_coef": out},
              open(FINAL / f"m5r_single_stage_auxdecode{tag}.json", "w"), indent=2)
    print(f"\n[single-stage] === summary (refs: {REFS[args.method]}) ===", flush=True)
    for coef, d in out.items():
        print(f"  aux_coef={coef:>4}: return {d['return_mean']:.1f}±{d['return_std']:.1f}  "
              f"belief-probe {d['probe_acc_mean']:.3f}±{d['probe_acc_std']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
