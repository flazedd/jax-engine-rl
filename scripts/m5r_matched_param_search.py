"""Re-equalise parameter counts for the matched-input variants.

The locked configs tuned `hidden_dim` per variant so every method lands near
the ~5,000-parameter budget. Matching the policy inputs changes the counts:
the RL² concat policy and critic now read [o_t, h_t] rather than h_t alone,
the hypernet target networks take the 11-dim base observation rather than the
16-dim augmented one, and the VariBAD hypernet gained a belief-only trunk.

This searches `hidden_dim` for each variant under the new architecture and
reports the setting closest to the budget, so the ablation varies the
conditioning architecture and not capacity.

Usage:
  uv run python -m scripts.m5r_matched_param_search
"""
from __future__ import annotations

import sys

import jax

from agents.rl2 import RL2Agent
from agents.varibad import VariBADAgent

OBS_SIZE = 16      # RL2ObsEnv: 11 inventory + 3 prev-action + reward + done
BASE_OBS = 11      # o_t, the policy-side observation under matched inputs
N_ACTIONS = 3
BUDGET = 5000


def _count(agent) -> int:
    params = agent.init(jax.random.PRNGKey(0))["params"]
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _rl2(integration: str, hidden: int):
    return RL2Agent(
        obs_size=OBS_SIZE, n_actions=N_ACTIONS, integration=integration,
        hidden_dim=hidden,
        hypernet_target_hidden=4, hypernet_hidden=8, hypernet_init_scale=0.01,
        policy_trunk_layers=0, policy_trunk_hidden=0,
        policy_obs_dim=BASE_OBS, concat_policy_reads_obs=True,
    )


def _varibad(integration: str, hidden: int):
    return VariBADAgent(
        obs_size=OBS_SIZE, n_actions=N_ACTIONS, integration=integration,
        hidden_dim=hidden, latent_dim=2, reward_decoder="gaussian",
        hypernet_target_hidden=4, hypernet_hidden=8, hypernet_init_scale=0.01,
        policy_trunk_layers=2, policy_trunk_hidden=hidden,
        policy_obs_dim=BASE_OBS,
    )


VARIANTS = [
    ("rl2_concat", lambda h: _rl2("concat", h)),
    ("rl2_hypernet", lambda h: _rl2("hypernet", h)),
    ("varibad_concat", lambda h: _varibad("concat", h)),
    ("varibad_hypernet", lambda h: _varibad("hypernet", h)),
]


def main() -> int:
    print(f"target budget: ~{BUDGET} trainable parameters\n")
    chosen = {}
    for name, build in VARIANTS:
        best = None
        for hidden in range(4, 65):
            try:
                n = _count(build(hidden))
            except Exception as e:  # architecture invalid at this width
                print(f"  [{name}] hidden={hidden} failed: {e}")
                continue
            if best is None or abs(n - BUDGET) < abs(best[1] - BUDGET):
                best = (hidden, n)
        chosen[name] = best
        hidden, n = best
        print(f"{name:18s} hidden_dim={hidden:3d}  params={n:5d}  "
              f"({n - BUDGET:+d} vs budget)")

    counts = [v[1] for v in chosen.values()]
    print(
        f"\nspread across variants: {min(counts)} to {max(counts)} "
        f"({max(counts) - min(counts)} apart, "
        f"{100 * (max(counts) - min(counts)) / BUDGET:.1f}% of budget)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
