"""Solve the matched-capacity widths for every method in a domain.

Fairness has three parts, and this script handles the third:
  1. Same inputs      — every method reads the per-step tuple
                        u_t = [o_t, a_{t-1}, r_{t-1}, d_{t-1}] (env wiring).
  2. Same optimiser   — epochs, minibatching, learning rate, clipping shared
                        across methods (config wiring).
  3. Same capacity    — every method within ~1% of one parameter budget, with
                        the two conditioning architectures of a method sharing
                        an identical belief encoder so only the conditioning
                        architecture differs. That is what this solves.

The encoder width is held common across a method's {concat, hypernet} pair and
the policy-side width is the free knob, so equal totals over identical encoders
leave the policy allocations equal too.

Usage:
  uv run python -m scripts.matched_budget_search              # both domains
  uv run python -m scripts.matched_budget_search --domain cartpole
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import jax

from agents.ppo import PPOAgent
from agents.rl2 import RL2Agent
from agents.varibad import VariBADAgent

BUDGET = 5000


@dataclass(frozen=True)
class Domain:
    name: str
    base_obs: int      # dim of o_t
    n_actions: int
    n_regimes: int = 3
    stack_k: int = 4

    @property
    def tuple_obs(self) -> int:
        """u_t = [o_t, one-hot a_{t-1}, r_{t-1}, d_{t-1}]."""
        return self.base_obs + self.n_actions + 2

    @property
    def regime_obs(self) -> int:
        """u_t plus the regime signal handed to Belief-PPO / Oracle-PPO."""
        return self.tuple_obs + self.n_regimes

    @property
    def stacked_obs(self) -> int:
        return self.stack_k * self.tuple_obs


DOMAINS = {
    "market_making": Domain("market_making", base_obs=11, n_actions=3),
    "cartpole": Domain("cartpole", base_obs=4, n_actions=2),
}


def _count(agent) -> int:
    tree = agent.init(jax.random.PRNGKey(0))["params"]
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


# ---------------------------------------------------------------------------
# Per-method builders. Every meta-RL builder takes (encoder_hidden, trunk) so
# the encoder can be pinned across a pair while the trunk absorbs the budget.
# ---------------------------------------------------------------------------


def _ppo(d: Domain, obs: int, hidden: int):
    return PPOAgent(obs_size=obs, n_actions=d.n_actions, hidden_dim=hidden)


def _rl2(d: Domain, integration: str, hidden: int, trunk: int):
    return RL2Agent(
        obs_size=d.tuple_obs, n_actions=d.n_actions, integration=integration,
        hidden_dim=hidden,
        hypernet_target_hidden=4, hypernet_hidden=12 if integration == "hypernet" else 0,
        hypernet_init_scale=0.01,
        policy_trunk_layers=1, policy_trunk_hidden=trunk,
        policy_obs_dim=d.base_obs, concat_policy_reads_obs=True,
    )


def _varibad(d: Domain, integration: str, hidden: int, trunk: int):
    return VariBADAgent(
        obs_size=d.tuple_obs, n_actions=d.n_actions, integration=integration,
        hidden_dim=hidden, latent_dim=2, reward_decoder="gaussian",
        hypernet_target_hidden=4, hypernet_hidden=4 if integration == "hypernet" else 0,
        hypernet_init_scale=0.01,
        policy_trunk_layers=2, policy_trunk_hidden=trunk,
        policy_obs_dim=d.base_obs,
    )


BUILDERS = {"rl2": _rl2, "varibad": _varibad}


def _best_monotone(count_at, lo: int, hi: int, budget: int) -> tuple[int, int]:
    """Width closest to `budget`, exploiting that params grow with width.

    Binary-searches the crossing point rather than enumerating the range, which
    keeps the pair search to a few hundred agent initialisations instead of
    tens of thousands.
    """
    cache: dict[int, int] = {}

    def f(x: int) -> int:
        if x not in cache:
            cache[x] = count_at(x)
        return cache[x]

    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) < budget:
            lo = mid + 1
        else:
            hi = mid
    candidates = [x for x in (lo - 1, lo, lo + 1) if x >= 1]
    scored = [(abs(f(x) - budget), x, f(x)) for x in candidates]
    _, x, n = min(scored)
    return x, n


def _solve_reference(d: Domain, obs: int, budget: int) -> tuple[int, int]:
    """Feedforward reference: hidden width is the only knob."""
    return _best_monotone(lambda h: _count(_ppo(d, obs, h)), 4, 128, budget)


def _solve_pair(d: Domain, method: str, budget: int) -> dict:
    """Common encoder width across the pair, policy trunk free per variant.

    Scored on the worse of the two variants, so the chosen encoder is the one
    at which *both* conditioning architectures can reach the budget.
    """
    build = BUILDERS[method]
    best = None
    for hidden in range(6, 41):
        per_variant = {}
        ok = True
        for integration in ("concat", "hypernet"):
            try:
                per_variant[integration] = _best_monotone(
                    lambda t: _count(build(d, integration, hidden, t)),
                    2, 96, budget,
                )
            except Exception:  # architecture invalid at this width
                ok = False
                break
        if not ok:
            continue
        worst = max(abs(v[1] - budget) for v in per_variant.values())
        # Tie-break toward the larger encoder: more belief capacity at equal fit.
        if best is None or (worst, -hidden) < (best[0], -best[1]):
            best = (worst, hidden, per_variant)
    worst, hidden, per_variant = best
    return {"hidden": hidden, "variants": per_variant, "worst_delta": worst}


def run_domain(d: Domain, budget: int) -> list[tuple[str, int]]:
    print(f"\n{'='*78}\n{d.name}  (o_t={d.base_obs}, u_t={d.tuple_obs}, "
          f"u_t+regime={d.regime_obs}, stacked={d.stacked_obs})\n{'='*78}")
    counts: list[tuple[str, int]] = []

    print(f"{'method':<24}{'setting':<38}{'params':>8}{'Δ':>7}")
    for label, obs in [
        ("regime-agnostic PPO", d.tuple_obs),
        ("stacked-observation PPO", d.stacked_obs),
        ("Belief-PPO", d.regime_obs),
        ("Oracle-PPO", d.regime_obs),
    ]:
        hidden, n = _solve_reference(d, obs, budget)
        print(f"{label:<24}{f'hidden_dim={hidden}':<38}{n:>8}{n-budget:>+7}")
        counts.append((label, n))

    for method in ("rl2", "varibad"):
        sol = _solve_pair(d, method, budget)
        for integration, (trunk, n) in sol["variants"].items():
            label = f"{method}-{integration}"
            setting = (f"hidden_dim={sol['hidden']}, "
                       f"policy_trunk_hidden={trunk}")
            print(f"{label:<24}{setting:<38}{n:>8}{n-budget:>+7}")
            counts.append((label, n))

    lo, hi = min(c for _, c in counts), max(c for _, c in counts)
    print(f"\nspread: {lo} to {hi}  ({hi - lo} apart, "
          f"{100 * (hi - lo) / budget:.1f}% of budget)")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.matched_budget_search")
    ap.add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
    ap.add_argument("--budget", type=int, default=BUDGET)
    args = ap.parse_args()

    domains = list(DOMAINS.values()) if args.domain == "all" else [DOMAINS[args.domain]]
    for d in domains:
        run_domain(d, args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
