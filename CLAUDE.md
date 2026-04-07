# CLAUDE.md — Bayesian RL for Market Making Under Hidden Regime Switching

> JAX-native (jit, vmap, lax.scan). All state/config objects: `typing.NamedTuple`. Stack: JAX + Equinox + Optax + Distrax.

---

## Thesis Framing

POMDP market-making environment with analytically tractable optimal policies as performance ceilings. Environment characterised analytically before any RL training — locked-regime and mixed-regime oracles computed, policies separated by true regime, value of optimal action selection quantified. This foundation validates the environment and establishes targets.

**Research questions (RQ2 is the centerpiece):**

> **RQ2** — Does explicit Bayesian inference produce superior and more sample-efficient market-making policies than implicit recurrent memory? *(main contribution: seven-gap decomposition with Belief-PPO ablation)*
>
> **RQ1** — Do meta-RL agents develop regime-structured internal representations from market observations alone? *(supporting mechanistic evidence)*
>
> **RQ3** *(exploratory)* — Does the timing of regime inference emergence during training predict downstream policy improvement?

---

## RQ2 — Policy Quality Decomposition (centerpiece)

Train the full agent ladder on the mixed-regime environment. The seven-gap decomposition is the central result:

```
Oracle A → Oracle B             cost of partial observability (env property)
Oracle B → Belief-PPO           cost of learning vs planning (approx policy opt)
Belief-PPO → VariBAD            cost of approximate vs exact Bayesian inference
VariBAD → RL²+HN                value of explicit posterior vs implicit HN conditioning
RL²+HN → RL²                   value of hypernetwork architecture (Beck et al. 2023)
RL² → PPO+LSTM                  value of meta-learning (multi-episode trials)
PPO+LSTM → PPO MLP              value of any memory at all
```

**Primary metrics:**
- **Normalised return:** `(agent - random) / (Oracle_B - random)`. Y-axis [-0.2, 1.1]. Oracle A sits above 1.0 as dashed reference.
- **AULC (Area Under Learning Curve):** integral of normalised return over training steps, divided by total steps. Range [0,1]. Primary sample efficiency metric — captures both speed and asymptotic level. Wilcoxon signed-rank across 8 seeds.
- **Steps to 0.8 × Oracle_B_return:** secondary threshold metric. Agents that never reach → "did not reach."
- **Per-regime action distributions** (Figure 6) — tests genuine regime-appropriate behaviour vs better compromise.
- **Wilcoxon signed-rank + Cohen's d** between each adjacent agent pair across 8 seeds.

**Figures:** 3, 4, 6

## RQ1 — Regime-Structured Representations (mechanistic)

At regular checkpoints, freeze encoder, extract latent representations over **held-out evaluation episodes**. Train supervised logistic regression probe.

**Probe protocol:** 256 held-out eval episodes per checkpoint (fixed eval seeds, separate from training). Collect `(latent, true_regime)` at every step. Stratified 80/20 split by regime. Train probe on 80%, report accuracy on 20% only.

**Representations probed:** GRU hidden state (RL², RL²+HN), posterior mean μ_t (VariBAD).

**Evidence required:** probe accuracy >> 33% chance; pairwise L2 centroid distances increasing over training; pattern consistent across seeds; corroborated by Figure 6 action distributions.

**Caveat:** probe measures linear decodability only. Always interpreted jointly with per-regime action distributions. High probe + near-diagonal matrix = regime inference confirmed. High probe + off-diagonal = encodes but doesn't act on regime (publishable). Low probe + good performance = non-linear/alternative solution (publishable).

**Figures:** 5, 7

## RQ3 — Inference-Performance Timing (exploratory)

Per agent per seed, identify: `t_infer` = step where probe accuracy first > 60%; `t_perf` = step where normalised return first > 0.6. Compute `t_infer - t_perf` across seeds. Sign test against zero.

If `t_infer < t_perf` consistently → inference leads performance (causal support). If non-significant → co-development is the defensible conclusion. Framed as exploratory — thesis stands on RQ2 regardless.

**Figures:** 5 (with markers)

---

## Environment Specification

### Regimes & HMM
```
3 regimes: noise(0), bull(1), bear(2)
HMM = [[0.95, 0.03, 0.02],
        [0.08, 0.90, 0.02],
        [0.08, 0.02, 0.90]]
Expected regime duration: ~12 steps. ~10 regime visits per T=200 episode.
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
drift_probs = [[0.15, 0.70, 0.15],  # noise — zero mean
               [0.05, 0.50, 0.45],  # bull  — positive drift
               [0.45, 0.50, 0.05]]  # bear  — negative drift
```

### Inventory & Reward
```
q' = clip(q + fill_bid - fill_ask, -5, 5)     # 11 levels
σ² = [0.5, 1.5, 1.5]                           # noise, bull, bear

r_t = fill_bid·δ_bid + fill_ask·δ_ask          # spread PnL (both sides earn)
      - 0.1 · q'² · σ²_regime                  # inventory risk (post-step q')
      - 5.0 · |q'| · 1(|q'|==5)               # boundary penalty

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
| Belief-PPO | `(o_t, b_t)` | MLP + exact HMM filter (true params) | Multi-episode trial (4 eps) | Perfect inference ablation |
| Oracle B | `(q, b)` | VI + exact HMM filter | — | POMDP optimality ceiling |
| Oracle A | `(q, r)` | VI lookup | — | Full-information upper bound |

### Multi-Episode Trial Structure
RL², RL²+HN, VariBAD, Belief-PPO all train on **4-episode trials** (800 steps). HMM initial regime sampled from stationary distribution at trial start, then runs normally. Hidden states persist across episode boundaries within a trial. PPO+LSTM also evaluated on 4-episode trials but its hidden state resets every episode — it cannot exploit cross-episode information.

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

### Shared Q_max
```
Q_max = max_{q, r} Q*_locked(q, r, a*(q,r))
```
Computed once from locked-regime VI. Single denominator for all relative gain (%) computations across all figures.

### Oracle A — Full-Information MDP (33 states)
```
V(q,r) = max_a { E[r(a,q,r)] + γ · Σ_{r'} P(r'|r) · Σ_{q'} P(q'|q,a,r) · V(q',r') }
P(q'|q,a,r): 4 outcomes from independent fills, then clip.
vmap over 33 states per sweep. Converge: max|V_new - V_old| < 1e-6.
Output: π*_A(q,r), Q*_A(q,r,a)
```

### Oracle B — POMDP Belief-State (~4400 states)
```
20×20 triangular grid over Δ².
Belief update: predict → likelihood-weight → normalise (exact HMM filter).
V(q,b) = max_a { E[r(a,q,b)] + γ · Σ_o P(o|a,b) · V(q'(o,a), b'(o,a,b)) }
Nearest-grid-point interpolation. vmap over all states per sweep.
Output: π*_B(q,b), Q*_B(q,b,a)
```

---

## Analytical Foundation — Preconditions

> Computed before any RL training. Gate: if any precondition fails → retune κ/σ² before proceeding.

### Precondition 1 — Locked-regime policies distinct and decisive
VI on each regime independently (11-state MDP). Extract π*_r(q), Q*_r(q,a).
```
gap_r(q) = (Q*_r(q,a*) - Q*_r(q,a_2nd)) / Q_max × 100%
disagreement(r1, r2) = mean_q[π*_r1(q) ≠ π*_r2(q)]
Pass: all disagreements > 20%, mean gap > 5% per regime.
```

### Precondition 2 — Oracle A regime-dependent under switching
Roll out Oracle A in switching env. Tag steps by true_regime_t.
```
Per true regime: action distribution, gap_A(q,r) on shared Q_max.
Pass: action distributions clearly separate by true regime; mean gap > 5%.
```

### Precondition 3 — Oracle B regime-dependent under belief uncertainty
Roll out Oracle B with exact HMM filter. Tag by true_regime_t.
```
Per true regime: action distribution, gap_B(q,r) averaged over encountered beliefs.
Pass: distributions separate despite uncertain beliefs; mean gap > 5%.
```

---

## Key Figures

### Figure 1 — Locked-Regime Oracle ★ FOUNDATION
2 rows × 3 cols (noise | bull | bear).
Row 1: optimal action heatmap per q. Row 2: relative gain bar chart (gap_r(q)).
Below: pairwise disagreement bars + numerical table (mean/min gap, % with gap>10%).

### Figure 2 — Mixed-Regime Oracles ★ FOUNDATION
4 rows × 3 cols (true regime).
Row 1: Oracle A actions by true regime. Row 2: Oracle A relative gain.
Row 3: Oracle B actions by true regime. Row 4: Oracle B relative gain.
Below: Oracle A vs locked disagreement; Oracle B vs Oracle A disagreement; pairwise disagreement comparison.

### Figure 3 — PPO Validation ★ IMPLEMENTATION CHECK
Learning curves. PPO locked-noise/bull/bear → ~1.0. PPO mixed → ~0.3–0.5.
Shading: 25th–75th percentile, 8 seeds. Reference: Oracle B=1.0, Oracle A>1.0, random=0.

### Figure 4 — Agent Ladder (RQ2)
Learning curves for all 5 RL agents. Reference lines for Oracle A, Oracle B, random.
Gap annotations (labeled brackets) per decomposition table.
Inset: AULC bar chart + threshold-crossing bars (steps to 0.8×Oracle_B).

### Figure 5 — Regime Inference Over Training (RQ1 + RQ3)
One panel per probed agent (RL², RL²+HN, VariBAD). Dual y-axis: probe accuracy (left), normalised return (right).
Threshold lines: 60% (t_infer), 0.6 (t_perf). Per-seed tick marks.
Below: distribution of t_infer − t_perf with sign test p-value. Framed as exploratory.

### Figure 6 — Per-Regime Action Distributions (RQ1 + RQ2)
3×3 normalised matrix per agent (rows=true regime, cols=action chosen).
Show Oracle A + all 5 RL agents + Belief-PPO. Visual progression from diagonal (oracle) to uniform (PPO MLP).

### Figure 7 — Latent Space Geometry (RQ1, supplementary)
2D PCA, one panel per probed agent × early/late training. Color by true regime.
Pairwise L2 centroid distances reported numerically.

---

## Implementation Phases

### Phase 0 — Environment Core
- `EnvState(NamedTuple)`: `(inventory, regime, mid_price, step)`
- `EnvParams(NamedTuple)`: all κ, δ, drift_probs, σ², HMM, γ, penalties — frozen
- HMM step: `jax.random.categorical` on transition row
- Fills: two independent `jax.random.bernoulli`
- Inventory: `jnp.clip(q + fill_bid - fill_ask, -5, 5)`
- Mid-price: draw `mid_change` from `drift_probs[regime]`
- Reward: spread PnL + inventory penalty + boundary (post-step q')
- `get_obs(state, mid_change)` → 4D
- `lax.scan` rollout returning `(obs, actions, rewards, true_regimes)`
- Multi-episode trial wrapper: 4 consecutive episodes (800 steps), HMM continuous across episodes, optional hidden-state reset flag per agent type

### Phase 1 — Locked-Regime VI + Oracle A
- Locked VI: per regime, fix regime, VI on 11-state MDP
- Oracle A: full HMM, 33-state MDP
- `vmap` over states per sweep. Converge: `< 1e-6`
- Extract Q-tables, policies, shared Q_max

### Phase 2 — Oracle B (POMDP VI)
- 20×20 triangular belief grid
- Exact HMM filter: predict, likelihood-weight, normalise
- Nearest-grid-point interpolation
- `vmap` over ~4400 states per sweep
- Extract π*_B, Q*_B

### Phase 3 — Analytical Foundation Figures
- Compute gap tables → Figure 1
- Oracle A rollout in switching env, tag by true_regime → Figure 2 rows 1–2
- Oracle B rollout with HMM filter → Figure 2 rows 3–4
- Pairwise disagreements, comparison summaries
- **Gate:** mean gap < 5% or any disagreement < 20% → halt, retune

### Phase 4 — PPO MLP Baseline
- MLP policy + value network. Input: o_t only
- GAE via `lax.scan`
- Locked-regime runs (3) + mixed-regime run
- Normalised return → Figure 3
- **Gate:** locked curves must reach ~1.0

### Phase 5 — PPO+LSTM
- LSTM policy, hidden state across episode via `lax.scan`
- Input: `(o_t, a_{t-1}, r_{t-1})`
- **Single-episode training** — h resets at every episode boundary
- Evaluated on 4-episode trials but h still resets each episode

### Phase 6 — RL²
- GRU meta-policy, input: `(o_t, a_{t-1}, r_{t-1})`
- **Multi-episode trials:** 4 episodes per trial, h persists across episode boundaries
- Policy head on GRU output

### Phase 7 — RL²+HN (Beck et al. 2023)
- Same GRU as Phase 6
- HyperNet: `h_t → all weights & biases of policy MLP`
- Policy MLP receives `o_t` as input (dual conditioning: via h_t through HN + direct)
- Bias-HyperInit: HN final layer zero weights, non-zero bias
- Multi-episode trials

### Phase 8 — VariBAD
- GRU encoder → h_t; MLP posterior → (μ_t, σ_t), latent dim=4
- MLP decoder: (z_t, o_t, a_t) → (r̂_t, ô_{t+1}) via reparameterisation
- MLP policy: (o_t, μ_t, σ_t) → action logits
- Loss: L_PPO + β·L_ELBO, β=1.0
- Multi-episode trials
- Log (μ_t, σ_t) at each step for RQ1 probe

### Phase 9 — Belief-PPO (ablation)
- Exact HMM filter with true env params → b_t ∈ Δ² at each step
- PPO MLP with input (o_t, b_t), dim=7
- Multi-episode trials
- Isolates: if this matches Oracle B → problem is pure policy learning; if VariBAD matches this → approximate inference is sufficient

### Phase 10 — Evaluation & Figures
- 8 seeds per agent; all agents evaluated on same env seeds
- Normalised return curves → Figure 4
- AULC computation (primary) + threshold-crossing (secondary)
- Wilcoxon + Cohen's d between adjacent pairs
- Logistic probe at each checkpoint: 256 held-out eval episodes, stratified 80/20 split, report test accuracy only → Figure 5
- t_infer/t_perf extraction, sign test → Figure 5 (exploratory)
- Per-regime action distributions → Figure 6
- PCA latent projections → Figure 7

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

# HMM
HMM_TRANSITION    = [[0.95, 0.03, 0.02],
                      [0.08, 0.90, 0.02],
                      [0.08, 0.02, 0.90]]

# Fill model
KAPPA             = [[2.0, 2.0],   # noise
                     [2.0, 0.8],   # bull
                     [0.8, 2.0]]   # bear

# Actions
DELTA             = [[1.0, 1.0],   # symmetric
                     [1.0, 3.0],   # lean-ask
                     [3.0, 1.0]]   # lean-bid

# Mid-price drift
DRIFT_PROBS       = [[0.15, 0.70, 0.15],
                     [0.05, 0.50, 0.45],
                     [0.45, 0.50, 0.05]]

# Volatility & reward
SIGMA_SQ          = [0.5, 1.5, 1.5]
GAMMA_INVENTORY   = 0.1
BOUNDARY_PENALTY  = 5.0

# Oracle B
BELIEF_GRID_SIZE  = 20

# VariBAD
VARIBAD_LATENT_DIM = 4
VARIBAD_BETA       = 1.0

# RL²+HN
HN_GRU_SIZE       = 128
HN_POLICY_HIDDEN  = 64           # generated policy MLP hidden size

# Training & evaluation
N_SEEDS           = 8
SAMPLE_EFF_TARGET = 0.8          # threshold fraction of Oracle_B return
N_EVAL_EPISODES   = 256          # for probe at each checkpoint
PROBE_TRAIN_FRAC  = 0.8          # stratified by regime
```