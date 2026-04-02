#!/usr/bin/env python3
"""Train an RL agent — per-regime and mixed-regime analysis.

Produces:
  plots/{agent}_per_regime.png   — trained on each locked regime separately
  plots/{agent}_mixed_regime.png — trained on mixed regimes, evaluated per regime

Usage:
  uv run python scripts/train.py                    # defaults to ppo
  uv run python scripts/train.py --agent ppo
  uv run python scripts/train.py --agent rl2        # (once implemented)
  uv run python scripts/train.py --fast             # quick smoke test
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from lob_sim.agents import available_agents, make_agent
from lob_sim.actions import N_ACTIONS, BID_TICKS, ASK_TICKS
from lob_sim.config import SimConfig
from lob_sim.obs import observe
from lob_sim.state import init_state
from lob_sim.step import make_step_fn
from lob_sim.training.eval import evaluate_agent
from lob_sim.training.rollout import collect_rollout_batch
from lob_sim.training.trainer import compute_gae, create_optimizer, ppo_update

from plot_style import (
    apply_style, action_heatmap, mark_optimal, save_fig,
    REGIME_NAMES, REGIME_COLORS, MIXED_COLOR, N_BID, N_ASK,
)

apply_style()


# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--agent", default="ppo", choices=available_agents(),
                    help="Agent type to train")
parser.add_argument("--fast", action="store_true",
                    help="Quick smoke test with minimal iterations")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

AGENT_NAME = args.agent
SIM_CFG = SimConfig()

MAX_ITERS      = 10  if args.fast else 500
EVAL_EVERY     = 5   if args.fast else 10
N_EVAL_EPISODES = 5  if args.fast else 30

# Convergence: stop when relative improvement over the last PATIENCE
# evals is below REL_THRESHOLD (fraction of |current mean|).
# With EVAL_EVERY=10, PATIENCE=10 compares the last 100 iters vs the
# previous 100 — wide enough to smooth out eval noise.
PATIENCE      = 10
REL_THRESHOLD = 0.02   # 2%
ABS_FLOOR     = 0.1    # minimum absolute delta to avoid stopping at reward ≈ 0


# ── Load MC optimal actions from Phase 4 ─────────────────────────
_mc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "plots", "mc_optimal.json")
if os.path.exists(_mc_path):
    with open(_mc_path) as _f:
        _mc = json.load(_f)
    MC_OPTIMAL = _mc["optimal_actions"]  # [noise_idx, bull_idx, bear_idx]
    print(f"  MC optimal actions loaded from {_mc_path}")
    from lob_sim.actions import ACTION_TABLE as _AT
    for _r, _a in enumerate(MC_OPTIMAL):
        print(f"    {REGIME_NAMES[_r]}: action {_a} = "
              f"({int(_AT[_a][0])},{int(_AT[_a][1])})")
else:
    MC_OPTIMAL = None
    print(f"  WARNING: {_mc_path} not found — run plot_divergence.py first")
    print(f"           Blue squares will mark agent's most frequent action instead")


# ── Helpers ──────────────────────────────────────────────────────
def collect_eval_actions(agent, sim_config, rng_key, n_episodes, locked_regime):
    """Run evaluation episodes and collect all actions taken."""
    step_fn = make_step_fn(sim_config, locked_regime=locked_regime)
    n_steps = sim_config.max_steps

    def run_one(key):
        k_init, k_agent, k_run = jax.random.split(key, 3)
        sim_state = init_state(sim_config, k_init)
        agent_state = agent.initial_agent_state(k_agent)

        def step(carry, _):
            s, a_st, rng = carry
            rng, rng_act = jax.random.split(rng)
            obs = observe(s, sim_config)
            action, new_a_st, _ = agent.get_action(obs, a_st, rng_act)
            new_s, sim_out = step_fn(s, action)
            new_a_st = new_a_st._replace(
                prev_reward=sim_out["reward"],
                prev_done=sim_out["done"].astype(jnp.float32),
            )
            return (new_s, new_a_st, rng), action

        _, actions = jax.lax.scan(
            step, (sim_state, agent_state, k_run), None, length=n_steps
        )
        return actions

    keys = jax.random.split(rng_key, n_episodes)
    return jax.vmap(run_one)(keys)


def action_freq_matrix(actions):
    """Flat action array → (N_BID, N_ASK) frequency matrix."""
    flat = np.array(actions).reshape(-1)
    counts = np.bincount(flat, minlength=N_ACTIONS)
    return (counts / counts.sum()).reshape(N_BID, N_ASK)


def freq_heatmap(ax, freq_matrix, title, vmax=None):
    if vmax is None:
        vmax = max(0.4, freq_matrix.max() * 1.1)
    return action_heatmap(ax, freq_matrix, title,
                          kind="frequency", vmin=0, vmax=vmax)


# ── Training loop (agent-agnostic) ──────────────────────────────
def train_agent(agent_name, locked_regime, seed=42):
    """Train until convergence or MAX_ITERS.

    Convergence: improvement over last PATIENCE evals < 2% of |current mean|.

    Returns (agent, config, eval_rewards, eval_iters, entropies).
    """
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)

    # Fast mode: smaller network, fewer envs for speed
    fast_overrides = dict(n_envs=16, n_steps=128) if args.fast else {}
    agent, cfg = make_agent(agent_name, key=k0, **fast_overrides)
    optimizer, opt_state = create_optimizer(cfg, agent)

    regime_str = REGIME_NAMES[locked_regime] if locked_regime >= 0 else "MIXED"

    eval_rewards, eval_iters, entropies = [], [], []
    for i in range(MAX_ITERS):
        key, k_roll, k_upd = jax.random.split(key, 3)
        batch = collect_rollout_batch(
            agent, SIM_CFG, k_roll,
            n_envs=cfg.n_envs, n_steps=cfg.n_steps,
            locked_regime=locked_regime,
        )
        adv, ret = compute_gae(batch, cfg.gamma, cfg.gae_lambda)
        agent, opt_state, metrics = ppo_update(
            agent, optimizer, opt_state, batch, adv, ret, cfg, k_upd,
        )

        if i % EVAL_EVERY == 0 or i == MAX_ITERS - 1:
            key, k_eval = jax.random.split(key)
            stats = evaluate_agent(
                agent, SIM_CFG, k_eval,
                n_episodes=N_EVAL_EPISODES,
                locked_regime=locked_regime if locked_regime >= 0 else -1,
            )
            eval_rewards.append(float(stats["mean_reward"]))
            eval_iters.append(i)
            entropies.append(float(metrics["entropy"]))
            print(f"  [{regime_str}] iter {i:3d} | "
                  f"reward: {eval_rewards[-1]:7.3f} | "
                  f"entropy: {entropies[-1]:.3f}")

            # Convergence check
            n = len(eval_rewards)
            if n >= 2 * PATIENCE:
                prev_mean = sum(eval_rewards[n - 2*PATIENCE : n - PATIENCE]) / PATIENCE
                curr_mean = sum(eval_rewards[n - PATIENCE :]) / PATIENCE
                delta = curr_mean - prev_mean
                threshold = max(abs(curr_mean) * REL_THRESHOLD, ABS_FLOOR)
                if delta < threshold:
                    print(f"  [{regime_str}] converged at iter {i} "
                          f"(Δ {delta:.2f} < {threshold:.2f} "
                          f"[{REL_THRESHOLD:.0%} of |{curr_mean:.2f}|])")
                    break

    return agent, cfg, eval_rewards, eval_iters, entropies


# ── Print run config ─────────────────────────────────────────────
import sys
_cmd = " ".join(sys.argv)
print("=" * 60)
print(f"  {_cmd}")
print("=" * 60)
print(f"  agent:        {AGENT_NAME}")
print(f"  seed:         {args.seed}")
print(f"  fast:         {args.fast}")
print(f"  max_iters:    {MAX_ITERS}")
print(f"  eval_every:   {EVAL_EVERY}")
print(f"  eval_episodes:{N_EVAL_EPISODES}")
print(f"  convergence:  patience={PATIENCE}, "
      f"threshold={REL_THRESHOLD:.0%} relative, "
      f"floor={ABS_FLOOR}")
print("=" * 60)


# ── Part 1: Per-regime training ──────────────────────────────────
print(f"\n  Part 1: Training {AGENT_NAME.upper()} separately on each regime")
print("-" * 60)

per_regime_agents = {}
per_regime_freqs = {}
per_regime_data = {}

for regime_idx in range(3):
    name = REGIME_NAMES[regime_idx]
    print(f"\n--- {name} regime ---")
    agent, cfg, rewards, iters, entropies = train_agent(
        AGENT_NAME, regime_idx, seed=args.seed)
    per_regime_agents[regime_idx] = agent
    per_regime_data[regime_idx] = (rewards, iters, entropies)

    key_eval = jax.random.PRNGKey(99 + regime_idx)
    actions = collect_eval_actions(agent, SIM_CFG, key_eval,
                                   n_episodes=N_EVAL_EPISODES,
                                   locked_regime=regime_idx)
    per_regime_freqs[regime_idx] = action_freq_matrix(actions)

# Plot
plt.close("all")
fig1 = plt.figure(figsize=(18, 13))
gs1 = GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35,
               width_ratios=[1.2, 1.2, 1.0])

for regime_idx in range(3):
    name = REGIME_NAMES[regime_idx]
    color = REGIME_COLORS[regime_idx]
    rewards, iters, entropies = per_regime_data[regime_idx]

    ax = fig1.add_subplot(gs1[regime_idx, 0])
    ax.plot(iters, rewards, color=color)
    ax.set_title(f"{name} — Reward")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Eval Reward")
    ax.set_xlim(0, max(iters)); ax.grid(True)

    ax = fig1.add_subplot(gs1[regime_idx, 1])
    ax.plot(iters, entropies, color=color)
    ax.set_title(f"{name} — Entropy")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Entropy")
    ax.set_xlim(0, max(iters)); ax.grid(True)

    ax = fig1.add_subplot(gs1[regime_idx, 2])
    freq_heatmap(ax, per_regime_freqs[regime_idx], f"{name} — Actions")
    _opt = MC_OPTIMAL[regime_idx] if MC_OPTIMAL else int(np.argmax(per_regime_freqs[regime_idx]))
    mark_optimal(ax, _opt, origin="upper")

fig1.suptitle(f"{AGENT_NAME.upper()} Trained Per-Regime",
              fontweight="bold", y=0.98)
save_fig(fig1, f"{AGENT_NAME}_per_regime.png", script_file=__file__)
plt.close(fig1)


# ── Part 2: Mixed-regime training ────────────────────────────────
print(f"\n  Part 2: Training {AGENT_NAME.upper()} on mixed regimes")
print("-" * 60)

agent_mixed, cfg, rewards_mixed, iters_mixed, entropies_mixed = \
    train_agent(AGENT_NAME, -1, seed=args.seed + 81)

mixed_freqs = {}
for regime_idx in range(3):
    key_eval = jax.random.PRNGKey(200 + regime_idx)
    actions = collect_eval_actions(agent_mixed, SIM_CFG, key_eval,
                                   n_episodes=N_EVAL_EPISODES,
                                   locked_regime=regime_idx)
    mixed_freqs[regime_idx] = action_freq_matrix(actions)

# Plot
fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.40, wspace=0.35)

ax = fig2.add_subplot(gs2[0, :2])
ax.plot(iters_mixed, rewards_mixed, color=MIXED_COLOR)
ax.set_title(f"Mixed-Regime {AGENT_NAME.upper()} — Learning Curve")
ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Eval Reward"); ax.grid(True)

ax = fig2.add_subplot(gs2[0, 2])
ax.plot(iters_mixed, entropies_mixed, color=MIXED_COLOR, alpha=0.6)
ax.set_title("Entropy")
ax.set_xlabel("Iteration"); ax.set_ylabel("Entropy"); ax.grid(True)

vmax_global = max(max(f.max() for f in mixed_freqs.values()) * 1.1, 0.4)
for regime_idx in range(3):
    ax = fig2.add_subplot(gs2[1, regime_idx])
    freq_heatmap(ax, mixed_freqs[regime_idx],
                 f"Eval on {REGIME_NAMES[regime_idx]}", vmax=vmax_global)
    _opt = MC_OPTIMAL[regime_idx] if MC_OPTIMAL else int(np.argmax(per_regime_freqs[regime_idx]))
    mark_optimal(ax, _opt, origin="upper")

fig2.suptitle(f"{AGENT_NAME.upper()} on Mixed Regimes — Action Distribution per Regime\n"
              "(blue squares = Monte Carlo optimal action per regime)",
              fontweight="bold")
save_fig(fig2, f"{AGENT_NAME}_mixed_regime.png", script_file=__file__)
plt.close(fig2)


# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

print(f"\nPer-regime trained {AGENT_NAME.upper()} — most frequent action:")
for regime_idx in range(3):
    freq = per_regime_freqs[regime_idx]
    best = np.argmax(freq)
    bid = BID_TICKS[best // len(ASK_TICKS)]
    ask = ASK_TICKS[best % len(ASK_TICKS)]
    print(f"  {REGIME_NAMES[regime_idx]:6s}: bid={bid}, ask={ask}  "
          f"({freq.max()*100:.1f}% of actions)")

print(f"\nMixed-regime {AGENT_NAME.upper()} — most frequent action per eval regime:")
for regime_idx in range(3):
    freq = mixed_freqs[regime_idx]
    best = np.argmax(freq)
    bid = BID_TICKS[best // len(ASK_TICKS)]
    ask = ASK_TICKS[best % len(ASK_TICKS)]
    print(f"  {REGIME_NAMES[regime_idx]:6s}: bid={bid}, ask={ask}  "
          f"({freq.max()*100:.1f}% of actions)")

mixed_best = [np.argmax(mixed_freqs[r]) for r in range(3)]
n_distinct = len(set(mixed_best))
if n_distinct == 1:
    print(f"\n>>> {AGENT_NAME.upper()} plays the SAME action in all regimes — compromise policy!")
elif n_distinct < 3:
    print(f"\n>>> {AGENT_NAME.upper()} partially differentiates "
          f"({n_distinct}/3 distinct actions)")
else:
    print(f"\n>>> {AGENT_NAME.upper()} uses different actions per regime!")
