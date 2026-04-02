# JAX LOB Simulator + RL Agents

## Commands
- Always `uv run` (never bare python/pytest): `uv run pytest tests/test_phaseN.py -v`
- Install: `uv sync`

## --fast Flag
Scripts >30s support `--fast` (<10s). Use it first to catch bugs.
- `train.py --agent ppo --fast` (10 iter)
- `plot_divergence.py --fast` (10ep/200steps)

## Overview
JAX-native LOB simulator with HMM regimes → train PPO, RL², VariBAD, VariBAD+HyperNet to adapt to regime switches. Complete each phase before starting next.

## Global Constraints
- Python 3.12, `uv`, JAX only (no PyTorch/TF)
- Pure functions over JAX arrays, no Python control flow in JIT (`jnp.where`/`lax.scan`/`lax.cond`)
- Static array shapes, CPU-first (Apple Silicon), GPU-compatible
- `typing.NamedTuple` for state/config (pytree-compatible)

## Global Pitfalls
1. **RNG reuse**: split upfront, never reuse keys
2. **`jnp.roll` wrap**: mask stale values: `jnp.where(jnp.arange(n) >= n - shift, 0.0, rolled)`
3. **No Python if/for in JIT**: use `jnp.where`, `lax.scan`, `lax.fori_loop`
4. **Div-by-zero**: `jnp.where(denom > 0, num / denom, 0.0)`
5. **Action space = 3×3 (9)**: `BID_TICKS=[1,3,5]`, `ASK_TICKS=[1,3,5]`. Use `N_ACTIONS` from `lob_sim.actions`

## Dependencies
Phase 1–5: `jax jaxlib matplotlib pytest`
Phase 6+: add `optax distrax equinox`
Phase 10: add `scipy scikit-learn`

## Architecture
```
lob_sim/{__init__,config,regime,state,matching,background,obs,reward,actions,step}.py
lob_sim/agents/{__init__,base,networks,ppo,rl2,varibad,varibad_hyper}.py
lob_sim/training/{__init__,rollout,trainer,vae_buffer,eval,logger}.py
tests/test_phase{1..10}.py
scripts/{plot_sanity,plot_regimes,plot_divergence,plot_style,train}.py
```

## Scripts
| Script | Purpose |
|---|---|
| `train.py` | Train any agent: `--agent ppo\|rl2\|varibad\|varibad_hyper`. Per-regime + mixed training with convergence detection. Outputs `plots/{agent}_per_regime.png`, `plots/{agent}_mixed_regime.png` |
| `plot_sanity.py` | Phase 2 sim engine sanity check → `plots/sanity.png` |
| `plot_regimes.py` | Phase 3 regime dynamics visualization → `plots/regimes.png` |
| `plot_divergence.py` | Phase 4 MC policy divergence proof → `plots/divergence.png` |
| `plot_style.py` | Shared visual style (palette, heatmaps, `mark_optimal()`, `save_fig()`). Imported by all plot scripts |

## Adding a New Agent
1. Implement `lob_sim/agents/{name}.py` following the `Agent` protocol in `base.py`
2. Register in `lob_sim/agents/__init__.py`: add a `_build_{name}` function and `_REGISTRY["{name}"] = _build_{name}`
3. Run: `uv run python scripts/train.py --agent {name}`
4. Same training loop, convergence, eval, and plots — only the agent differs
5. **Note**: `train.py` currently calls `ppo_update` directly. Agents needing a different update (e.g. RL² needs sequential GRU, VariBAD needs dual optimizer) will need update-function dispatch added to the training loop

---

# PHASE 1 — Core Book Mechanics

Files: `lob_sim/{__init__,config,state,matching,background}.py`, `tests/test_phase1.py`

### SimConfig (NamedTuple)
n_levels=100, tick_size=0.02, limit_order_rate=0.25, limit_order_size=1.0, market_buy_prob=0.20, market_sell_prob=0.20, market_order_size_min=1.0, market_order_size_max=8.0, cancel_prob=0.10, depth_decay=0.05, price_drift=0.0, volatility_scale=1.0, agent_order_size=1.0, max_inventory=20, inventory_penalty=0.0002, max_steps=1000, initial_volume_per_level=1.5, initial_spread_ticks=4

### OrderBookState (NamedTuple)
Fields: bid_volumes(n_levels,), ask_volumes(n_levels,), mid_price(scalar f32), half_spread_ticks(scalar i32), regime(scalar i32, =0 for now), inventory(scalar f32), cash(scalar f32), step_count(scalar i32), done(scalar bool), rng_key(PRNGKey)

Price: `bid[i] = mid - (i + half_spread_ticks) * tick_size`, `ask[i] = mid + (i + half_spread_ticks) * tick_size`

`init_state(config, rng_key)`: volume = `initial_volume_per_level * exp(-depth_decay * i)`, mid=100.0

### matching.py
`fill_market_buy(ask_volumes, qty, half_spread_ticks, mid_price, tick_size)`:
1. cumsum ask_volumes → consumed[i] = min(vol[i], max(0, qty - cumsum[i-1]))
2. VWAP from consumed × level_prices
3. Find new best ask via `argmax(new_volumes > 0)`, roll+zero-fill deep end, update mid
4. Return: (new_ask_volumes, filled_qty, avg_fill_price, new_mid_price, new_half_spread)

`fill_market_sell(...)`: mirror on bid side.

### background.py
`generate_background_flow(bid_volumes, ask_volumes, config, rng_key)` — split 10+ subkeys
1. Cancellations: per-level prob cancel_prob, cancel 0–50% fraction
2. New limits: rate `limit_order_rate * exp(-depth_decay * i)`, Bernoulli
3. Market orders: buy prob market_buy_prob, sell prob market_sell_prob, size uniform [min,max]
Return: (new_bid_vols, new_ask_vols, mkt_buy_qty, mkt_sell_qty)

### Tests (test_phase1.py)
- TestInitState: volumes decreasing, mid=100, shapes=(n_levels,), all fields are jax arrays
- TestMatchingEngine: partial fill, full depletion, multi-level, mid_price shifts, VWAP correct, roll zero-fill, sell mirrors buy, zero qty noop, jit compatible
- TestBackgroundFlow: market order rate ~15%, cancellations reduce volume, limits add more near than deep, different keys→different results, jit compatible

---

# PHASE 2 — Sim Loop + Sanity Plots

Files: `lob_sim/{obs,reward,step}.py`, `tests/test_phase2.py`, `scripts/plot_sanity.py`

### obs.py
`observe(state, config) → (3*OBS_DEPTH+3,)` where OBS_DEPTH=10 (=33 dims)
Concat: top-10 bid vols (norm), top-10 ask vols (norm), cumulative imbalance per depth `(cum_bid-cum_ask)/(cum_bid+cum_ask+eps)`, spread norm, inventory norm, PnL proxy `(cash+inv*mid)/1000`. No regime.

### reward.py
`compute_reward(prev, next, config)` = value(next)-value(prev) - penalty*inv²
Step.py uses inline reward: `bid_fill*bid_edge + ask_fill*ask_edge + mtm - penalty*inv²`
where mtm = inv_before*(mid_after-mid_before). MTM is critical for regime-differentiated rewards.

### step.py
`make_step_fn(config) → step_fn(state, action) → (state, output_dict)`
Phase 2 action = (2,) array [bid_offset, ask_offset]. Steps:
1. Place agent orders 2. Snapshot pre-match 3. Background flow 4. Match market orders
5. Agent fills pro-rata (div-by-zero guard) 6. Remove unfilled 7. Obs+reward
8. Output: obs,reward,mid_price,spread,inventory,cash,bid_fill,ask_fill,done,bid_volumes[:20],ask_volumes[:20]
9. Done if abs(inv)>=max_inventory or step>=max_steps

`run_episode(config, rng_key, actions)`: actions shape (T,2), uses lax.scan

### plot_sanity.py → plots/sanity.png
3×2: mid-price, spread, LOB depth snapshot, LOB heatmap, inventory, cumulative reward

### Tests (test_phase2.py, T=2000)
- TestObservation: shape, obs_size=33, no NaN, jit
- TestReward: zero when unchanged, positive when profitable, inventory penalty
- TestStepFunction: jit compiles, step_count increments, agent orders removed, done on max_inventory
- TestRunEpisode: lax.scan compiles, mid_price not flat (std>0.001), fills happen, inventory fluctuates, output shapes, throughput >5000 steps/sec

---

# PHASE 3 — Regimes + Discrete Actions

Files: `lob_sim/{regime,actions}.py` (new), modify `background.py`+`step.py`, `tests/test_phase3.py`, `scripts/plot_regimes.py`

### regime.py
NOISE=0, BULL=1, BEAR=2, N_REGIMES=3
Transition: NOISE→[0.98,0.01,0.01], BULL→[0.02,0.97,0.01], BEAR→[0.02,0.01,0.97]

RegimeStepParams (per-regime, indexed by regime):
| Param | Noise | Bull | Bear |
|---|---|---|---|
| market_buy_prob | 0.15 | 0.30 | 0.05 |
| market_sell_prob | 0.15 | 0.05 | 0.30 |
| price_drift | 0.0 | +0.003 | -0.003 |
| volatility_scale | 1.0 | 1.3 | 1.3 |
| cancel_prob | 0.10 | 0.12 | 0.12 |
| limit_order_rate | 0.25 | 0.20 | 0.20 |

`transition_regime(current, rng_key)` via `jax.random.choice`
`get_regime_params(regime) → RegimeStepParams` via array indexing

### actions.py
BID_TICKS=[1,3,5], ASK_TICKS=[1,3,5] → ACTION_TABLE shape (9,2) via itertools.product, N_ACTIONS=9
Helpers: `action_index_to_offsets(idx)`, `offsets_to_action_index(bid, ask)`

### step.py modifications
- `make_step_fn(config, locked_regime=-1)`: action is scalar int32 → lookup ACTION_TABLE[action]
- Step 1: transition regime, override via `jnp.where` if locked, get regime_params
- Step 2: `mid_price += regime_params.price_drift`
- Background gets regime_params; market sizes × volatility_scale
- Output includes "regime"
- `run_episode(config, rng_key, actions, locked_regime=-1)`: actions (T,) int32

### plot_regimes.py → plots/regimes.png
2×2: mid-price per regime, cum fills per regime, inventory per regime, avg LOB depth per regime

### Tests (test_phase3.py)
- TestRegime: rows sum to 1, >90% stay NOISE over 1000, all 3 reachable over 10000, params correct, jit
- TestActions: shape (9,2), contents [1,1]..[5,5], round-trip
- TestLockedRegime: locked_noise stays 0, locked_bull stays 1, unlocked switches over T=5000
- TestRegimeEffectOnDynamics: bull drifts up, bear drifts down, noise no drift, bull more ask fills, bear more bid fills, run_episode with discrete action

---

# PHASE 4 — Policy Divergence Proof

Files: `scripts/plot_divergence.py`, `tests/test_phase4.py`

MC eval: 9 actions × 3 regimes, N=200 episodes, T=1000 steps, locked regime, fixed action, vmap over episodes.

### Tests (test_phase4.py, N=100, T=500 for speed)
- reward_matrix shape (N_ACTIONS, 3)
- **Optimal actions differ** for each regime (len(set)==3)
- Noise optimal: symmetric (|bid-ask|≤1)
- Bull optimal: bid < ask (tight bid, wide ask)
- Bear optimal: bid > ask (wide bid, tight ask)
- Cross-regime penalty: wrong policy worse than right policy

### plot_divergence.py → plots/divergence.png (N=200, T=1000)
Three 3×3 heatmaps, optimal per regime, cross-regime penalty table, Cohen's d

---

# PHASE 5 — Performance Benchmarks

File: `tests/test_phase5.py` (T=5000)
- Single env >10,000 steps/sec (warm)
- vmap 64 envs compiles, shape (64,T)
- Batch scaling: batch 64 > 2× batch 1 throughput
- No OOM at batch 1024 (T=1000)

---

# PHASE 6 — Agent Framework + PPO

Files: `lob_sim/agents/{__init__,base,networks,ppo}.py`, `lob_sim/training/{__init__,rollout,trainer,eval,logger}.py`, `scripts/train.py`, `tests/test_phase6.py`

### base.py
```
AgentState(NamedTuple): hidden(hidden_size,), prev_action(i32), prev_reward(f32), prev_done(f32)=0.0
RolloutBatch(NamedTuple): obs(n_envs,n_steps,obs_dim), actions(n_envs,n_steps), log_probs, values, rewards, dones, last_value(n_envs,)
Agent(Protocol): initial_agent_state(key)→AgentState, get_action(obs,state,key)→(action,state,info), update(batch,opt_state,key)→(agent,opt_state,metrics)
```

### networks.py (equinox)
- MLP: Linear layers with ReLU, final no activation
- GRUCell: wraps eqx.nn.GRUCell, (input, hidden) → new_hidden

### ppo.py
PPOConfig: lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, n_epochs=4, n_minibatches=4, n_envs=64, n_steps=256, hidden_size=0

PPOAgent(eqx.Module): trunk MLP(33→[64,64]→64), policy_head Linear(64→N_ACTIONS), value_head Linear(64→1), ppo_config=static
- initial_agent_state: all zeros (stateless)
- get_action: forward → logits,value; sample via distrax.Categorical; return (action, unchanged_state, {log_prob, value})

### rollout.py — Closed-Loop (agent picks actions inside lax.scan)
DO NOT use run_episode. New scan with RolloutCarry(sim_state, agent_state, rng_key).
```
collect_rollout(agent, sim_config, rng_key, n_steps, locked_regime=-1, meta_episode=False)
  → (final_carry, StepOutput{obs,action,log_prob,value,reward,done}, last_value)
collect_rollout_batch(..., n_envs, ...) → RolloutBatch  # vmap over envs
```

### trainer.py
`compute_gae(batch, gamma, gae_lambda)` → (advantages, returns) via reverse lax.scan
- delta_t = r_t + γ*V(t+1)*(1-done_t) - V(t); A_t = delta_t + γ*λ*(1-done_t)*A_{t+1}
- returns = advantages + values

`ppo_update(agent, optimizer, opt_state, batch, adv, ret, ppo_config, key)`:
- Flatten (n_envs,n_steps)→(total,), shuffle, split into minibatches
- PPO loss: clipped ratio × advantages + value_coef*MSE + entropy_coef*entropy
- Optimizer: `optax.chain(clip_by_global_norm(0.5), adam(lr))`

`create_optimizer(ppo_config, agent) → (optimizer, opt_state)`
`rl2_ppo_update(...)`: sequential through GRU, minibatches over envs not shuffled time

### eval.py
`evaluate_agent(agent, sim_config, key, n_episodes=50, locked_regime=-1) → {mean_reward, std_reward, mean_episode_length, rewards}`

### logger.py
Simple Python class: log(iter, metrics), print_summary(last_n), get_means(key, last_n)

### train.py
`uv run python scripts/train.py --agent ppo [--fast] [--seed 42]`
Trains per-regime (3 locked regimes) then mixed. Convergence-based stopping (PATIENCE=5, <2% relative improvement, max 500 iters). Produces `plots/{agent}_per_regime.png` and `plots/{agent}_mixed_regime.png`.

### Tests (test_phase6.py, <60s total)
FAST_PPO_CONFIG: n_epochs=2, n_minibatches=2, n_envs=16, n_steps=128
- TestAgentInterface: initial_state shape, get_action shapes+range [0,N_ACTIONS), jit, stateless, different keys→different actions
- TestRollout: single compiles, shapes (N,33)/(N,), batch shapes (n_envs,n_steps,...), actions valid, no NaN, throughput >5000/s
- TestGAE: shapes, no NaN, known trajectory [1.5,0.5,0.5] with γ=λ=1, returns=adv+values
- TestPPOUpdate: runs+finite metrics, params change, entropy>0
- TestTrainingImprovement: 30 iters, last_10 reward >= first_10 - 20% tolerance

---

# PHASE 7 — RL² Agent

Files: `lob_sim/agents/rl2.py`, `tests/test_phase7.py`

Architecture: `[obs(33); prev_action one-hot(N_ACTIONS); prev_reward(1)] → GRU(128) → hidden → policy_head(N_ACTIONS) / value_head(1)`
RL2AgentState: GRU hidden(128,), prev_action, prev_reward
Meta-episode: n_episodes_per_meta_episode=3, env resets but hidden persists

### Tests
- hidden shape (128,), hidden changes after get_action, input dim=33+N_ACTIONS+1+1, jit, hidden persists across done
- Trains 50 iters without error, outperforms PPO on mixed regimes, hidden correlates with regime

---

# PHASE 8 — VariBAD

Files: `lob_sim/agents/varibad.py`, `lob_sim/training/vae_buffer.py`, `tests/test_phase8.py`

Encoder: (s_t, a_{t-1}, r_t) → FC → GRU(64) → linear → (μ, log_σ) ∈ ℝ^5
Decoder: (s_t, a_t, m) → MLP[64,32] → predicted r_{t+1}
Policy: [obs(33); μ(5); σ(5)] → MLP[64,64] → policy(N_ACTIONS) / value(1)
Training: separate optimizers — PPO lr=7e-4 for policy, Adam lr=1e-3 for VAE. RL loss doesn't backprop through encoder. VAE from replay buffer. ELBO at 8 subsampled timesteps.

### Tests
- Initial posterior = prior N(0,I), posterior updates after observations, policy input dim=43, jit, reparameterization
- Buffer: add+sample, max_size
- VAE reconstruction loss decreases, KL>0.1 (no collapse), latents cluster by regime (L2>0.1 pairwise), outperforms PPO, separate optimizers

---

# PHASE 9 — VariBAD + HyperNetwork

Files: `lob_sim/agents/varibad_hyper.py`, add HyperNetwork to `networks.py`, `tests/test_phase9.py`

HyperNet: m∈ℝ^5 → ReLU → h(m) = policy weights φ. Separate heads for actor/critic. Linear (single layer).
Bias-HyperInit: hypernetwork.bias = flatten(base_params), weight ≈ 0 → h(m) ≈ f for any m at init.

### Tests
- Output shapes match base network, bias-hyperinit (max diff <0.1), different m→different weights
- get_action jit, generated policy valid (softmax sums to 1)
- 100 iters no divergence, matches/beats VariBAD (≥0.9×), weight distance differs across regimes

---

# PHASE 10 — Comparison + Thesis Analysis

Files: `lob_sim/agents/oracle.py`, `scripts/{train_all_seeds,eval_agent,compare_agents,ablation_persistence}.py`, `tests/test_phase10.py`

### Oracle Agent
Plays MC-optimal action per true regime (from Phase 4). Theoretical ceiling.

### train_all_seeds.py
4 agents × 5 seeds (42–46) on mixed regimes, 1000 iterations → checkpoints/{agent}_{seed}.eqx

### compare_agents.py → plots/comparison.png
N=200 eval episodes per seed. 6-panel: bar chart w/ error bars + oracle line, per-regime breakdown, adaptation speed (windowed reward around transitions), learning curves (per 50 iters), VariBAD latent PCA, results table. Wilcoxon signed-rank p-values, Cohen's d.

### ablation_persistence.py → plots/ablation_persistence.png
3 settings: fast(0.90), medium(0.97), slow(0.995) self-transition. Evaluate trained agents. Plot VariBAD advantage vs persistence.

### Latent Regime Inference → plots/latent_inference_accuracy.png
Every 50 train iters: collect (μ, regime) pairs, logistic regression accuracy vs iteration.

### Tests (test_phase10.py)
- TestAgentRanking: VariBAD>PPO, VariBAD>PPO significant (p<0.05 Wilcoxon), VariBAD+HN>PPO, RL²>PPO, all below oracle (5% tolerance)
- TestPolicySpecialization: PPO uses compromise (≤2 modes), VariBAD specializes (bull≠bear mode), matches MC top-3
- TestAdaptationSpeed: VariBAD post-switch reward > PPO
- TestLatentRepresentation: regime classification >70% accuracy, mean latent L2 separation >0.5
- TestReproducibility: cross-seed CV <30% for all agents

---

# Running Tests
```bash
uv run pytest tests/ -v                    # all phases
uv run pytest tests/test_phase3.py -v      # specific phase
uv run pytest tests/test_phase5.py -v -s   # with output
uv run pytest tests/ -v -k "not phase4 and not phase10"  # skip slow
```
