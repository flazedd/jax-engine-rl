# CLAUDE.md — Bayesian RL for Market Making Under Hidden Regime Switching

> JAX-native (jit, vmap, lax.scan). All state/config objects: `typing.NamedTuple`. Stack: JAX + Equinox + Optax + Distrax.

---

## Thesis Framing

POMDP market-making environment with analytically tractable optimal policies as performance ceilings. Fixed regime per episode — each episode samples one regime and holds it constant. Agents must infer the regime from within-episode observations. Inventory resets between episodes. This is the textbook VariBAD setup: clean task identification over a meta-trial of 4 episodes. Per-regime optimal policies computed via locked-regime VI and verified to be distinct before any RL training.

**Research questions (RQ2 is the centerpiece):**

> **RQ2** — Does explicit Bayesian inference produce superior and more sample-efficient market-making policies than implicit recurrent memory? *(main contribution: seven-gap decomposition with Belief-PPO ablation)*
>
> **RQ1** — Do meta-RL agents develop regime-structured internal representations from market observations alone? *(supporting mechanistic evidence)*
>
> **RQ3** *(exploratory)* — Does the timing of regime inference emergence during training predict downstream policy improvement?

---

## RQ2 — Policy Quality Decomposition (centerpiece)

Train the full agent ladder on the fixed-regime-per-episode environment. The six-gap decomposition is the central result:

```
Oracle → Belief-PPO            cost of partial observability (inference transient)
Belief-PPO → VariBAD           cost of approximate vs exact Bayesian inference
VariBAD → RL²+HN               value of explicit posterior vs implicit HN conditioning
RL²+HN → RL²                   value of hypernetwork architecture (Beck et al. 2023)
RL² → PPO+LSTM                  value of meta-learning (multi-episode trials)
PPO+LSTM → PPO MLP              value of any memory at all
```

**Primary metrics:**
- **Mean reward per step:** raw metric, no normalisation. Y-axis in reward/step units. Reference line for Oracle (locked-regime), PPO MLP asymptotic. All agents compared on the same scale.
- **AULC (Area Under Learning Curve):** integral of mean reward/step over training steps, divided by total steps. Primary sample efficiency metric — captures both speed and asymptotic level. Wilcoxon signed-rank across 8 seeds.
- **Steps to 0.8 × Oracle reward/step:** secondary threshold metric. Agents that never reach → "did not reach."
- **Per-regime action distributions** (Figure 15) — tests genuine regime-appropriate behaviour vs better compromise.
- **Wilcoxon signed-rank + Cohen's d** between each adjacent agent pair across 8 seeds.

**Figures:** 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15

## RQ1 — Regime-Structured Representations (mechanistic)

At regular checkpoints, freeze encoder, extract latent representations over **held-out evaluation episodes**. Train supervised logistic regression probe.

**Probe protocol:** 256 held-out eval episodes per checkpoint (fixed eval seeds, separate from training). Collect `(latent, true_regime)` at every step. Stratified 80/20 split by regime. Train probe on 80%, report accuracy on 20% only.

**Representations probed:** GRU hidden state (RL², RL²+HN), posterior mean μ_t (VariBAD).

**Evidence required:** probe accuracy >> 33% chance; pairwise L2 centroid distances increasing over training; pattern consistent across seeds; corroborated by Figure 15 action distributions.

**Caveat:** probe measures linear decodability only. Always interpreted jointly with per-regime action distributions. High probe + near-diagonal matrix = regime inference confirmed. High probe + off-diagonal = encodes but doesn't act on regime (publishable). Low probe + good performance = non-linear/alternative solution (publishable).

**Figures:** 14, 16

## RQ3 — Inference-Performance Timing (exploratory)

Per agent per seed, identify: `t_infer` = step where probe accuracy first > 60%; `t_perf` = step where mean reward/step first > 0.8 × Oracle reward/step. Compute `t_infer - t_perf` across seeds. Sign test against zero.

If `t_infer < t_perf` consistently → inference leads performance (causal support). If non-significant → co-development is the defensible conclusion. Framed as exploratory — thesis stands on RQ2 regardless.

**Figures:** 14 (with markers)

---

## Environment Specification

### Regimes (fixed per episode)
```
3 regimes: noise(0), bull(1), bear(2)
Regime sampled uniformly at episode start, held fixed for the entire episode.
Inventory resets to 0 at each episode boundary.
In a 4-episode meta-trial, each episode independently samples its regime.
Agents must infer the regime from within-episode observations (fills, drift).
```

### Actions
```
3 actions: delta[action] = (δ_bid, δ_ask)
  0: symmetric  (1, 1)  — optimal in noise
  1: lean-ask   (1, 3)  — optimal in bull
  2: lean-bid   (3, 1)  — optimal in bear
```

### Fill Model
```
P(fill_side) = Bernoulli(exp(-κ[regime, side] · δ[action, side]))
fills are independent per side

κ = [[2.0, 2.0],   # noise — symmetric
     [2.0, 0.8],   # bull  — ask fills easily (informed buyers)
     [0.8, 2.0]]   # bear  — bid fills easily (informed sellers)
```

### Mid-Price Drift
```
mid_change ~ Categorical({-1, 0, +1})
drift_probs = [[0.20, 0.60, 0.20],  # noise — zero mean
               [0.12, 0.50, 0.38],  # bull  — positive drift
               [0.38, 0.50, 0.12]]  # bear  — negative drift

Per-step drift likelihood ratio ~1.9× (reduced from 3×) to force
evidence accumulation over multiple steps for confident regime ID.
```

### Inventory & Reward
```
q' = clip(q + fill_bid - fill_ask, -5, 5)     # 11 levels
σ² = [0.5, 1.5, 1.5]                           # noise, bull, bear

r_t = fill_bid·δ_bid + fill_ask·δ_ask          # spread PnL (both sides earn)
      - 0.04 · q'² · σ²_regime                 # inventory risk (post-step q')
      - 5.0 · |q'| · 1(|q'|==5)               # boundary penalty
      + 1.0 · q · mid_change                    # mark-to-market (pre-step q)

Low inventory penalty (0.04) lets regime exploitation dominate at most
inventory levels. Mark-to-market rewards holding inventory in the
direction of the regime drift. E[MTM|bull,q] = q·0.26, E[MTM|noise,q] = 0.
Agents that identify the regime early can accumulate directional inventory
and profit from the drift for the remainder of the episode.

γ_disc = 0.99, T_episode = 200
```

### Observations & Inputs
```
o_t = (fill_bid ∈ {0,1}, fill_ask ∈ {0,1}, mid_change ∈ {-1,0,+1}, q ∈ [-5,5])

PPO MLP:      o_t only (clean memoryless baseline)
All recurrent + Belief-PPO: (o_t, a_{t-1}, r_{t-1})
```

---

## Agent Ladder

| Agent | Input | Architecture | Training regime | What it tests |
|---|---|---|---|---|
| PPO MLP | `o_t` | MLP | Single episode | Memoryless baseline |
| PPO+LSTM | `(o_t, a_{t-1}, r_{t-1})` | LSTM, h resets each episode | Single episode | Implicit memory, no meta-learning |
| RL² | `(o_t, a_{t-1}, r_{t-1})` | GRU, h persists across episodes in trial | Multi-episode trial (4 eps) | End-to-end meta-RL |
| RL²+HN | `(o_t, a_{t-1}, r_{t-1})` | GRU → HyperNet → policy MLP; state re-conditioned | Multi-episode trial (4 eps) | Recurrent hypernetwork (Beck et al. 2023) |
| VariBAD | `(o_t, a_{t-1}, r_{t-1})` | GRU encoder → VAE posterior; policy on `(o_t, μ_t, σ_t)` | Multi-episode trial (4 eps) | Explicit approximate Bayesian inference |
| Belief-PPO | `(o_t, b_t)` | MLP + exact Bayesian filter (true params) | Multi-episode trial (4 eps) | Perfect inference ablation |
| Oracle | `(q, r)` | VI lookup (locked-regime) | — | Full-information upper bound |

### Multi-Episode Trial Structure
RL², RL²+HN, VariBAD, Belief-PPO all train on **4-episode trials** (800 steps). Each episode independently samples a regime (uniform) and holds it fixed. Inventory resets to 0 at each episode boundary. Hidden states persist across episode boundaries within a trial — enabling cross-episode meta-learning. PPO+LSTM also evaluated on 4-episode trials but its hidden state resets every episode — it cannot exploit cross-episode information.

### RL²+HN Architecture (Beck et al. 2023)
```
GRU: (o_t, a_{t-1}, r_{t-1}) → h_t (size 128)
HyperNet: h_t → all weights & biases of policy MLP
Policy MLP: o_t → action logits (generated by HyperNet)
  — state o_t is re-passed as input to generated policy (dual conditioning)
Init: Bias-HyperInit — HN final layer: zero weight matrix, non-zero bias
  → all trajectories produce identical base-network at init (stable training)
```

### VariBAD Architecture
```
Encoder:  GRU over (o_t, a_{t-1}, r_{t-1}) → h_t
Posterior: MLP(h_t) → (μ_t, σ_t), latent dim d=4
Decoder:  MLP(z_t, o_t, a_t) → (r̂_t, ô_{t+1}), z_t ~ N(μ_t, σ_t²)
Policy:   MLP(o_t, μ_t, σ_t) → action logits
Loss:     L_PPO + β·L_ELBO, β=1.0
          L_ELBO = E[log P(r_t, o_{t+1} | z_t, o_t, a_t)] - KL[q(z|h_t) || N(0,I)]
```

### Belief-PPO (new ablation)
```
Exact HMM filter with true (κ, drift_probs, HMM) parameters.
Input: (o_t, b_t) where b_t ∈ Δ² is the exact posterior over 3 regimes.
Architecture: standard PPO MLP, input dim = 4 (obs) + 3 (belief) = 7.
Trains with PPO on multi-episode trials.
Isolates: value of perfect Bayesian inference given learned policy.
```

---

## Oracle Specification

### Oracle — Locked-Regime VI (11 states per regime)
```
Per regime r, solve the 11-state MDP (inventory q ∈ [-5, 5]):
V_r(q) = max_a { E[r(a,q,r)] + γ · Σ_{q'} P(q'|q,a,r) · V_r(q') }
P(q'|q,a,r): 4 outcomes from independent fills, then clip.
Converge: max|V_new - V_old| < 1e-6.
Output: π*_r(q), Q*_r(q,a) for each regime r.

Q_max = max_{q, r} Q*_r(q, a*(q,r))
```
Single denominator for all relative gain (%) computations. The oracle knows the regime and uses the optimal locked-regime policy from step 1 — the upper bound for any agent.

---

## Analytical Foundation

> Computed before any RL training. Gate: if precondition fails → retune κ/σ²/γ_inventory before proceeding.

### Precondition — Locked-regime policies distinct and decisive
VI on each regime independently (11-state MDP). Extract π*_r(q), Q*_r(q,a).
```
gap_r(q) = (Q*_r(q,a*) - Q*_r(q,a_2nd)) / Q_max × 100%
disagreement(r1, r2) = mean_q[π*_r1(q) ≠ π*_r2(q)]
Pass: all disagreements > 20%, mean gap > 5% per regime.
```
Output: single figure with 3 heatmaps (noise | bull | bear), x=inventory, y=action, cell=optimal %.

---

## Key Figures

### Figure 1 — Per-Regime Optimal Policies ★ FOUNDATION
1 row × 3 cols (noise | bull | bear).
Each panel: heatmap (x=inventory q, y=action, cell=100% if optimal).
Pairwise disagreement percentages in title. Gate: PASS/FAIL.

### Figure 4 — PPO Policies ★ IMPLEMENTATION CHECK
3×3 grid: optimal vs PPO isolated vs PPO mixed action distributions per regime.
Row 0: Optimal policy. Row 1: PPO trained in isolation. Row 2: PPO mixed split by regime.

### Figure 5 — PPO Learning Curves ★ IMPLEMENTATION CHECK
Learning curves (reward/step). PPO locked regimes converge near Oracle for that regime.
PPO mixed converges well below Oracle — memoryless baseline cannot exploit regime structure.
Shading: 25th–75th percentile, 8 seeds. Reference line: Oracle (reward/step).

### Figure 6 — PPO+LSTM Policies ★ IMPLEMENTATION CHECK
3×3 grid: optimal vs PPO+LSTM isolated vs PPO+LSTM mixed action distributions per regime.
Row 0: Optimal policy. Row 1: PPO+LSTM trained in isolation. Row 2: PPO+LSTM mixed split by regime.

### Figure 7 — PPO+LSTM Learning Curves ★ IMPLEMENTATION CHECK
Learning curves (reward/step). PPO+LSTM locked regimes converge near Oracle for that regime.
PPO+LSTM mixed should outperform PPO MLP mixed — LSTM enables within-episode regime identification.
Shading: 25th–75th percentile, 8 seeds. Reference line: Oracle (reward/step).

### Figure 8 — PPO MLP vs PPO+LSTM vs RL² vs RL²+HN Comparison ★ IMPLEMENTATION CHECK
Learning curves overlaid: PPO MLP (blue) vs PPO+LSTM (orange) vs RL² (green) vs RL²+HN (red) per regime.
Top row: isolated. Bottom row: mixed split by regime. Oracle reference line.
Key comparison: RL²+HN mixed should outperform RL² mixed — HyperNet architecture enables richer conditioning.

### Figure 9 — RL² Policies ★ IMPLEMENTATION CHECK
3×3 grid: optimal vs RL² isolated vs RL² mixed action distributions per regime.
Row 0: Optimal policy. Row 1: RL² trained in isolation. Row 2: RL² mixed split by regime.
RL² trains on 4-episode trials with persistent GRU state across episode boundaries.

### Figure 10 — RL² Learning Curves ★ IMPLEMENTATION CHECK
Learning curves (reward/step). RL² locked regimes converge near Oracle for that regime.
RL² mixed should outperform PPO+LSTM mixed — cross-episode meta-learning enables faster regime ID.
Shading: 25th–75th percentile, 8 seeds. Reference line: Oracle (reward/step).

### Figure 11 — RL²+HN Policies ★ IMPLEMENTATION CHECK
3×3 grid: optimal vs RL²+HN isolated vs RL²+HN mixed action distributions per regime.
Row 0: Optimal policy. Row 1: RL²+HN trained in isolation. Row 2: RL²+HN mixed split by regime.
RL²+HN trains on 4-episode trials with HyperNet-generated policy from GRU hidden state.

### Figure 12 — RL²+HN Learning Curves ★ IMPLEMENTATION CHECK
Learning curves (reward/step). RL²+HN locked regimes converge near Oracle for that regime.
RL²+HN mixed should outperform RL² mixed — HyperNet enables richer regime conditioning.
Shading: 25th–75th percentile, 8 seeds. Reference line: Oracle (reward/step).

### Figure 13 — Agent Ladder (RQ2)
Learning curves (reward/step) for all RL agents. Reference lines for Oracle, PPO MLP asymptotic.
Gap annotations (labeled brackets) per decomposition table.
Inset: AULC bar chart + threshold-crossing bars (steps to 0.8×Oracle reward/step).

### Figure 14 — Regime Inference Over Training (RQ1 + RQ3)
One panel per probed agent (RL², RL²+HN, VariBAD). Dual y-axis: probe accuracy (left), reward/step (right).
Threshold lines: 60% (t_infer), 0.8×Oracle reward/step (t_perf). Per-seed tick marks.
Below: distribution of t_infer − t_perf with sign test p-value. Framed as exploratory.

### Figure 15 — Per-Regime Action Distributions (RQ1 + RQ2)
3×3 normalised matrix per agent (rows=true regime, cols=action chosen).
Show Oracle + all RL agents + Belief-PPO. Visual progression from diagonal (oracle) to uniform (PPO MLP).

### Figure 16 — Latent Space Geometry (RQ1, supplementary)
2D PCA, one panel per probed agent × early/late training. Color by true regime.
Pairwise L2 centroid distances reported numerically.

---

## Implementation Phases

### Phase 0 — Environment Core
- `EnvState(NamedTuple)`: `(inventory, regime, mid_price, step)`
- `EnvParams(NamedTuple)`: all κ, δ, drift_probs, σ², γ, penalties — frozen
- Regime: fixed per episode, sampled uniformly at episode start
- Fills: two independent `jax.random.bernoulli`
- Inventory: `jnp.clip(q + fill_bid - fill_ask, -5, 5)`, resets to 0 between episodes
- Mid-price: draw `mid_change` from `drift_probs[regime]`
- Reward: spread PnL - inventory penalty - boundary (post-step q') + MTM (pre-step q × mid_change)
- `get_obs(state, mid_change)` → 4D
- `lax.scan` rollout returning `(obs, actions, rewards, true_regimes)`
- Multi-episode trial wrapper: 4 consecutive episodes (800 steps), each episode samples regime independently, inventory resets at episode boundary, optional hidden-state reset flag per agent type

### Phase 1 — Locked-Regime VI (Oracle)
- Per regime, fix regime, VI on 11-state MDP
- `vmap` over states per sweep. Converge: `< 1e-6`
- Extract Q-tables, policies, shared Q_max

### Phase 2 — Analytical Foundation
- Compute gap tables and pairwise disagreements → Figure 1
- **Gate:** mean gap < 5% or any disagreement < 20% → halt, retune

### Phase 3 — PPO MLP Baseline
- MLP policy + value network. Input: o_t only
- GAE via `lax.scan`
- Locked-regime runs (3) + mixed-regime run
- Reward/step learning curves → Figure 5
- Per-regime action distributions → Figure 4
- **Gate:** locked curves must converge near respective Oracle reward/step

### Phase 4 — PPO+LSTM
- LSTM policy, hidden state across episode via `lax.scan`
- Input: `(o_t, a_{t-1}, r_{t-1})`
- **Single-episode training** — h resets at every episode boundary
- Evaluated on 4-episode trials but h still resets each episode

### Phase 5 — RL²
- GRU meta-policy, input: `(o_t, a_{t-1}, r_{t-1})`
- **Multi-episode trials:** 4 episodes per trial, h persists across episode boundaries
- Policy head on GRU output

### Phase 6 — RL²+HN (Beck et al. 2023)
- Same GRU as Phase 6
- HyperNet: `h_t → all weights & biases of policy MLP`
- Policy MLP receives `o_t` as input (dual conditioning: via h_t through HN + direct)
- Bias-HyperInit: HN final layer zero weights, non-zero bias
- Multi-episode trials

### Phase 7 — VariBAD
- GRU encoder → h_t; MLP posterior → (μ_t, σ_t), latent dim=4
- MLP decoder: (z_t, o_t, a_t) → (r̂_t, ô_{t+1}) via reparameterisation
- MLP policy: (o_t, μ_t, σ_t) → action logits
- Loss: L_PPO + β·L_ELBO, β=1.0
- Multi-episode trials
- Log (μ_t, σ_t) at each step for RQ1 probe

### Phase 8 — Belief-PPO (ablation)
- Exact Bayesian filter with true env params → b_t ∈ Δ² at each step
- PPO MLP with input (o_t, b_t), dim=7
- Multi-episode trials
- Isolates: if VariBAD matches this → approximate inference is sufficient

### Phase 9 — Evaluation & Figures
- 8 seeds per agent; all agents evaluated on same env seeds
- Reward/step learning curves → Figure 13
- AULC computation (primary) + threshold-crossing (secondary)
- Wilcoxon + Cohen's d between adjacent pairs
- Logistic probe at each checkpoint: 256 held-out eval episodes, stratified 80/20 split, report test accuracy only → Figure 14
- t_infer/t_perf extraction, sign test → Figure 14 (exploratory)
- Per-regime action distributions → Figure 15
- PCA latent projections → Figure 16

---

## Parameter Summary

```python
# Environment
N_REGIMES         = 3
N_ACTIONS         = 3
INVENTORY_BOUNDS  = (-5, 5)       # 11 levels
T_EPISODE         = 200
GAMMA_DISC        = 0.99
EPISODES_PER_TRIAL = 4            # for meta-learning agents
REGIME_SAMPLING   = "uniform"     # each episode samples regime independently

# Fill model
KAPPA             = [[2.0, 2.0],   # noise
                     [2.0, 0.8],   # bull
                     [0.8, 2.0]]   # bear

# Actions
DELTA             = [[1.0, 1.0],   # symmetric
                     [1.0, 3.0],   # lean-ask
                     [3.0, 1.0]]   # lean-bid

# Mid-price drift (reduced informativeness per step)
DRIFT_PROBS       = [[0.20, 0.60, 0.20],
                     [0.12, 0.50, 0.38],
                     [0.38, 0.50, 0.12]]

# Volatility & reward
SIGMA_SQ          = [0.5, 1.5, 1.5]
GAMMA_INVENTORY   = 0.04
BOUNDARY_PENALTY  = 5.0
MTM_WEIGHT        = 1.0           # mark-to-market: mtm_weight · q_pre · mid_change

# VariBAD
VARIBAD_LATENT_DIM = 4
VARIBAD_BETA       = 1.0

# RL²+HN
HN_GRU_SIZE       = 128
HN_POLICY_HIDDEN  = 64           # generated policy MLP hidden size

# Training & evaluation
N_SEEDS           = 8
SAMPLE_EFF_TARGET = 0.8          # threshold fraction of Oracle reward/step
N_EVAL_EPISODES   = 256          # for probe at each checkpoint
PROBE_TRAIN_FRAC  = 0.8          # stratified by regime
```