# Implementation reference

Full repo layout, interface contracts (`Agent`, `Env`, `Hypernet`, `ExplorationBonus`), JSON schemas for results, plotting module organization, and known implementation pitfalls.

Read this when writing or refactoring code — agents, envs, training loops, hypernet/exploration-bonus modules, result parsing, or plotting.

For JAX-specific performance rules, see `docs/conventions.md` → "JAX performance discipline" (they're tightly tied to general conventions and live there).

See also:
- `docs/conventions.md` — uv, YAML, git, tests, figure style.
- `docs/methodology.md` — the statistical comparison module `evaluation/comparisons.py`.

---

## Repo layout

This is the complete set of files the project will contain when M0–M6 are finished. Optional files (M7 ablations, scope-permitting figures) are marked.

```
thesis/
├── agents/
│   ├── __init__.py
│   ├── base.py                      # abstract Agent class, shared interface
│   ├── dummy.py                     # random-policy agent (M0 pipeline validation only)
│   ├── ppo.py                       # vanilla PPO (MLP). Used as-is for the regime-agnostic
│   │                                #   floor AND, wrapped with stack_obs, for stacked-obs PPO.
│   ├── ppo_oracle.py                # PPO conditioned on true regime one-hot
│   ├── ppo_belief.py                # PPO conditioned on analytical HMM posterior
│   ├── ppo_per_regime.py            # PPO trained on locked single regime — env-validation tool only (R2 check); not a method in the ladder or a reference level
│   ├── rl2.py                       # RL² (recurrent PPO with prev action/reward)
│   ├── varibad.py                   # VariBAD (variational belief + PPO + ELBO)
│   └── modules/                     # composable ablation components (not standalone agents)
│       ├── __init__.py
│       ├── hypernet.py              # hypernetwork integration: belief → policy weights.
│       │                            #   Composes with rl2.py or varibad.py via config flag.
│       └── exploration_bonus.py     # belief-novelty auxiliary reward. Novelty signal source
│                                    #   is method-specific (RL² hidden state, VariBAD mu).
│
├── envs/
│   ├── __init__.py
│   ├── base.py                      # abstract Env class, JAX-compatible interface
│   ├── mm_reduced.py                # main env: reduced-form MM. Accepts a regime_config.
│   │                                #   Supports E0 (no regimes), E1+ (regime-switching),
│   │                                #   and lock_regime: int | None for per-regime PPO.
│   ├── regime_bandit.py             # Markov-switching bandit (M4 validation + shared toy)
│   ├── validation/                  # implementation-validation envs (not thesis-relevant)
│   │   ├── __init__.py
│   │   ├── dummy.py                 # constant-reward env (M0 pipeline validation)
│   │   ├── bandit.py                # 2-armed Bernoulli bandit (M4 RL²/VariBAD sanity)
│   │   └── gridworld.py             # random-goal gridworld (M4)
│   └── wrappers/
│       ├── __init__.py
│       └── stack_obs.py             # observation-stacking wrapper; stacked-obs PPO = ppo.py + this
│
├── beliefs/
│   ├── __init__.py
│   ├── hmm_posterior.py             # analytical forward algorithm for HMM belief
│   └── oracle.py                    # true regime extraction from env_state.
│                                    #   Separated from env to enforce leak discipline: only
│                                    #   Oracle-PPO, Belief-PPO, per-regime PPO may import this.
│
├── oracles/
│   ├── __init__.py
│   ├── analytical_as.py             # closed-form AS optimal return (M1 target)
│   ├── value_iteration.py           # VI on full-info MDP (used by R1, R2, M3 sanity checks)
│   └── verify_requirements.py       # R1–R4 verification script (M2 main tool). CLI:
│                                    #   uv run python -m oracles.verify_requirements --env-config ...
│
├── training/
│   ├── __init__.py
│   ├── train.py                     # main entry point; dispatches by agent/env in config
│   ├── rollout.py                   # lax.scan rollout utilities
│   ├── ppo_update.py                # shared PPO loss/update. Used by every PPO-based agent
│   │                                #   including RL² — RL²'s loss is ordinary PPO loss,
│   │                                #   it differs only in architecture (recurrent) and data
│   │                                #   handling (trajectory-level minibatches). Branches on
│   │                                #   is_recurrent=True to preserve temporal order.
│   ├── varibad_update.py            # VariBAD-specific update: PPO loss + ELBO with two
│   │                                #   optimizers on two parameter groups (encoder/decoder
│   │                                #   vs policy). Internally calls ppo_update for the PPO
│   │                                #   portion. Needs its own file because the joint
│   │                                #   optimization structure is genuinely different.
│   └── config.py                    # dataclass configs per experiment + apply_run_mode()
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # gap-closed fractions, regime classification accuracy,
│   │                                #   factorial marginal means
│   ├── comparisons.py               # paired Wilcoxon + Holm correction + bootstrap CI;
│   │                                #   produces the "supported / not supported" decisions
│   ├── posterior_compare.py         # symmetric KL vs analytical HMM posterior, via linear
│   │                                #   probe to simplex (probe frozen after M5; see
│   │                                #   Statistical methodology)
│   └── posterior_performance.py     # M6 decoupling analysis: scatter of gap-closed vs
│                                    #   posterior error, per-method Spearman correlations
│
├── utils/
│   ├── __init__.py
│   └── script_output.py             # write_summary() helper; every script calls at end.
│                                    #   Produces the shared-schema JSON and OK/FAIL stdout line.
│
├── plotting/
│   ├── __init__.py
│   ├── style.py                     # apply_style(): colors, fonts, sizes. Called by every plot.
│   ├── load_results.py              # parse .json result files, handle missing fields
│   ├── regenerate_figures.py        # regenerate all figures for a milestone from existing JSONs.
│   │                                #   Called by scripts/make_milestone.py; usable standalone:
│   │                                #     uv run python -m plotting.regenerate_figures M{n}
│   ├── learning_curves.py           # training curves with CI bands (RQ1, RQ2, M1, M2, etc.)
│   ├── gap_decomposition.py         # fig_rq1_ceilings_bar, fig_rq1_gap_fractions
│   ├── posterior_quality.py         # fig_rq2_posterior_error, fig_M4_varibad_posterior_sharpening
│   ├── factorial.py                 # fig_rq2_factorial (2×2×2 grouped bars)
│   ├── difficulty_sweep.py          # fig_rq3_persistence_sweep, fig_rq3_distinguishability_sweep
│   └── posterior_vs_performance.py  # fig_rq3_posterior_vs_performance (decoupling scatter)
│
├── experiments/
│   └── configs/
│       ├── base/                    # composable base configs (hyperparameter discipline)
│       │   ├── base_ppo.yaml        # core PPO hyperparameters, tuned in M1, frozen thereafter
│       │   ├── base_rl2.yaml        # extends base_ppo with RL² specifics
│       │   ├── base_varibad.yaml    # extends base_ppo with VariBAD specifics
│       │   └── base_env_mm.yaml     # shared MM env parameters
│       ├── envs/                    # env-specific configs for M2 iteration
│       │   ├── e0_as_baseline.yaml
│       │   ├── e1_vol_switched.yaml
│       │   ├── e2_fill_switched.yaml
│       │   └── e_final.yaml         # symlink to whichever env version passed R1–R4
│       ├── m0_dummy.yaml            # M0 pipeline test
│       ├── m1_ppo_as.yaml           # M1: PPO on AS baseline
│       ├── m3_oracle.yaml
│       ├── m3_belief.yaml
│       ├── m3_regime_agnostic.yaml
│       ├── m4_rl2_bandit.yaml       # M4: validation, one file per (method, task) pair
│       ├── m4_rl2_gridworld.yaml
│       ├── m4_rl2_regime_bandit.yaml
│       ├── m4_varibad_bandit.yaml
│       ├── m4_varibad_gridworld.yaml
│       ├── m4_varibad_regime_bandit.yaml
│       ├── m4_ablation_hypernet.yaml       # hypernet vs concat on toys
│       ├── m4_ablation_exploration.yaml    # bonus vs no-bonus on toys
│       ├── m5_ladder_ppo.yaml       # M5 core ladder: one file per ladder rung
│       ├── m5_ladder_stacked_ppo.yaml
│       ├── m5_ladder_rl2.yaml
│       ├── m5_ladder_varibad.yaml
│       ├── m5_ladder_belief_ppo.yaml
│       ├── m5_ladder_oracle_ppo.yaml
│       ├── m5_factorial_rl2_concat_bonus.yaml          # M5 factorial (6 extra cells;
│       ├── m5_factorial_rl2_hypernet_nobonus.yaml      #   rl2_concat_nobonus and
│       ├── m5_factorial_rl2_hypernet_bonus.yaml        #   varibad_concat_nobonus overlap
│       ├── m5_factorial_varibad_concat_bonus.yaml      #   with m5_ladder_rl2 and
│       ├── m5_factorial_varibad_hypernet_nobonus.yaml  #   m5_ladder_varibad, don't re-run)
│       ├── m5_factorial_varibad_hypernet_bonus.yaml
│       ├── m5_mu_only_varibad_hypernet.yaml  # M5 mu-only vs full-posterior ablation
│       ├── m6_persistence_sweep.yaml     # M6: 3 points × methods × seeds
│       ├── m6_distinguishability_sweep.yaml
│       └── m6_heatmap.yaml               # optional 3×3 grid
│
├── tests/
│   ├── __init__.py
│   ├── test_envs.py                 # env invariants (inventory bounds, rewards finite, etc.)
│   ├── test_beliefs.py              # HMM posterior matches brute-force on short sequences
│   ├── test_oracles.py              # VI convergence, policy stability
│   ├── test_leak.py                 # regression: regime doesn't leak into non-oracle agents
│   ├── test_run_modes.py            # --super-fast < 30s, --fast < 5min
│   ├── test_script_output.py        # every write_summary JSON validates against schema
│   └── test_jax_perf.py             # second iteration doesn't recompile; catches shape drift
│
├── scripts/
│   ├── __init__.py
│   ├── make_milestone.py            # orchestrate a milestone end-to-end: run training,
│   │                                #   dump JSONs, generate plots, check artifacts,
│   │                                #   verify pass criteria. Called as:
│   │                                #     uv run python -m scripts.make_milestone M{n}
│   ├── run_ladder.py                # batch runner for M5 ladder (invokes train over
│   │                                #   each method config, then calls make_milestone M5)
│   └── run_sweep.py                 # M6 orchestration: run a grid of (difficulty × method
│                                    #   × seed) configs and aggregate into stats_M6_sweep.json
│
├── results/                         # gitignored; generated outputs
│   ├── milestones/                  # milestone-level artifacts
│   │   └── M{n}/
│   │       ├── stats_*.json         # verification JSONs per milestone
│   │       ├── PASS.md | FAIL.md    # review note, committed
│   │       └── ...
│   └── {experiment_name}/           # per-experiment outputs
│       ├── config.json              # effective config (run-mode applied), with commit hash
│       ├── summary.json             # shared-schema script output
│       ├── metrics.json             # per-iteration training metrics
│       ├── eval.json                # final evaluation numbers
│       └── checkpoints/             # optional saved params (flax serialization)
│
├── figures/                         # committed; PNGs referenced by LaTeX
│   ├── milestones/                  # milestone-internal figures
│   │   └── M{n}/
│   │       └── fig_M{n}_*.png
│   └── thesis/                      # figures cited in the thesis LaTeX
│       ├── fig_rq1_*.png
│       ├── fig_rq2_*.png
│       └── fig_rq3_*.png
│
├── pyproject.toml                   # uv-managed, exact version pins
├── uv.lock                          # committed lockfile for reproducibility
├── .gitignore                       # at minimum: results/, __pycache__/, .venv/, *.pyc
├── README.md                        # project summary + how to reproduce M0
└── CLAUDE.md                        # this file — the project spec
```

### Optional / scope-permitting files

These appear in the layout only if the corresponding scope decision goes "yes":

- `agents/modules/sequential_vs_set_encoder.py` — if the sequential-vs-set-encoder question gets tested as a mini-ablation in the discussion chapter.
- `envs/mm_variants.py` — if RQ3 ablations need env variants beyond the persistence / distinguishability sweep (e.g., different regime counts).
- `oracles/belief_vi.py` — VI on the belief-state POMDP (Oracle B). The current thesis framing uses Belief-PPO (analytical posterior + PPO) as the inferred-belief ceiling; Oracle B would be a stronger ceiling if included, but adds implementation complexity. Skip unless Belief-PPO underperforms significantly and you need a tighter ceiling.
- `experiments/configs/m7_*.yaml` — M7 supplementary ablations (stop-gradient, episode length, hidden-state probe). One file per ablation.

### Key structural points

**No HyperX, PEARL, classifier-head, or `_hn` files.** Hypernet integration and exploration bonus are composable modules in `agents/modules/`, selected via config flags on `rl2.py` and `varibad.py`. No `rl2_hn.py`, `varibad_hn.py`, `hyperx.py`, or `pearl.py`.

**Stacked-obs PPO = PPO + wrapper.** No separate `ppo_stacked.py`. The ladder rung "stacked-obs PPO" is `ppo.py` running on an env wrapped by `envs/wrappers/stack_obs.py`, selected via config.

**Per-regime PPO is not a reference level or method-ladder rung.** `agents/ppo_per_regime.py` is kept as an env-validation tool used in M2 to verify R2 (locked-regime optimality against VI). It is not evaluated in M3, M5, or M6.

**All composition is in configs, not code.** Whether an RL² run uses hypernet or concat, bonus or no bonus, is a config value. The same `rl2.py` file serves all factorial cells.

**One update file per distinct loss structure, not per agent.** `ppo_update.py` covers every agent whose loss is pure PPO: vanilla PPO, Oracle-PPO, Belief-PPO, *and RL²*. RL² differs from vanilla PPO in architecture (recurrent) and minibatching (trajectory-level to preserve temporal order), not in the loss function — so it uses the same update file with `is_recurrent=True`. Only VariBAD has a different loss structure (joint PPO + ELBO with two optimizers on two parameter groups), so only VariBAD gets its own `varibad_update.py`. Hypernet and exploration bonus don't warrant their own update files either: hypernet adds parameters that are trained through the same backward pass, and exploration bonus adds an extra reward term that flows through normal advantage computation.

**Milestone artifacts have two homes.** `results/milestones/M{n}/` for JSONs and `figures/milestones/M{n}/` for PNGs. Thesis figures are additionally copied to `figures/thesis/` under the `fig_rqN_*.png` name.


---

## Interface contracts

### `agents/base.py`

```python
class Agent:
    def init(self, key, obs_space, action_space, config) -> AgentState: ...
    def act(self, state, obs, key) -> tuple[action, new_state]: ...
    def update(self, state, trajectory_batch) -> tuple[new_state, metrics]: ...

    # Optional, implemented by belief-based agents:
    def infer_belief(self, state, history) -> Belief: ...

    # Optional metadata used by the training loop:
    requires_regime_label: bool = False  # only Oracle-PPO, per-regime PPO
    requires_analytical_posterior: bool = False  # only Belief-PPO
    is_recurrent: bool = False  # RL², VariBAD
    produces_belief_for_eval: bool = False  # all belief-based methods
```

`AgentState` is a pytree containing params, optimizer state, and any recurrent hidden state. Must be JAX-compatible (static shapes).

**Recurrent hidden state handling.** For recurrent agents (`is_recurrent == True`):
- The hidden state is part of `AgentState`, threaded through `act` calls.
- At episode boundaries, the hidden state is reset to the `init_hidden` value. This is the training loop's responsibility — the loop detects `done=True` and calls `reset_hidden_for_done` before the next `act`.
- RL² specifically augments input with `(obs, prev_action, prev_reward)`; this concatenation happens inside the agent, not in the env.

**Belief exposure for evaluation.** For agents with `produces_belief_for_eval == True`:
- `infer_belief(state, history)` returns the agent's current belief as a probability distribution over regimes (length-3 array summing to 1).
- The evaluation loop calls this on held-out trajectories to compute posterior-approximation error vs the analytical HMM posterior.
- Non-belief agents do not implement this method. PPO and stacked-obs PPO have no belief; the evaluation just skips them.

**Separate metrics for composite objectives.** Agents with auxiliary losses (VariBAD's ELBO, exploration bonus when enabled) return metrics dict with clear keys:
```python
{
    "ppo/policy_loss": ...,
    "ppo/value_loss": ...,
    "ppo/entropy": ...,
    "varibad/reconstruction_loss": ...,
    "varibad/kl": ...,
    "exploration_bonus/mean_bonus": ...,   # when exploration axis is on
}
```
Namespacing by component makes failure diagnosis trivial ("VariBAD's KL is blowing up" vs "VariBAD's return isn't improving" are different debugging paths).

### `agents/modules/hypernet.py`

The hypernet is a composable module that replaces the concat-integration path in RL² or VariBAD. Instead of the belief being concatenated to the observation and fed through a fixed policy network, the belief is fed to a hypernet that *generates the weights* of a small target policy network. The target network then maps observation to action logits using those belief-generated weights.

**Concrete shape** (flax):

```python
# agents/modules/hypernet.py
import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

class Hypernet(nn.Module):
    """Maps a belief vector to the weights of a small target policy network.

    The target network is a fixed 2-layer MLP: obs_dim → hidden → action_dim.
    Only the target weights (not its architecture) depend on the belief.
    """
    target_obs_dim: int
    target_hidden: int
    target_output_dim: int    # action_dim
    hypernet_hidden: int

    def target_param_count(self) -> int:
        """Total scalar weights needed to parameterize the target network."""
        return (
            self.target_obs_dim * self.target_hidden + self.target_hidden
            + self.target_hidden * self.target_output_dim + self.target_output_dim
        )

    @nn.compact
    def __call__(self, belief: jnp.ndarray) -> jnp.ndarray:
        """Belief → flat weight vector for the target network."""
        h = nn.Dense(self.hypernet_hidden)(belief)
        h = nn.relu(h)
        return nn.Dense(self.target_param_count())(h)

    def apply_target(self, flat_weights: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        """Unpack flat weights into target-network tensors and do forward pass."""
        i = 0
        W1 = flat_weights[i:i + self.target_obs_dim * self.target_hidden] \
                .reshape(self.target_obs_dim, self.target_hidden)
        i += self.target_obs_dim * self.target_hidden
        b1 = flat_weights[i:i + self.target_hidden]
        i += self.target_hidden
        W2 = flat_weights[i:i + self.target_hidden * self.target_output_dim] \
                .reshape(self.target_hidden, self.target_output_dim)
        i += self.target_hidden * self.target_output_dim
        b2 = flat_weights[i:i + self.target_output_dim]
        h = jnp.maximum(obs @ W1 + b1, 0.0)
        return h @ W2 + b2
```

**How agents use it.** Agents that support hypernet integration branch on `config.integration` in `setup`:

```python
# agents/varibad.py (sketch)
def setup(self, config):
    self.encoder = VariationalEncoder(...)
    self.decoder = Decoder(...)
    if config.integration == "concat":
        self.policy = PolicyMLP(input_dim=config.obs_dim + config.belief_dim, ...)
    elif config.integration == "hypernet":
        self.hypernet = Hypernet(
            target_obs_dim=config.obs_dim,
            target_hidden=config.hypernet_target_hidden,
            target_output_dim=config.action_dim,
            hypernet_hidden=config.hypernet_hidden,
        )

def act(self, state, obs, key):
    belief = self.encoder(state.trajectory_history)
    if self.config.integration == "concat":
        logits = self.policy(jnp.concatenate([obs, belief]))
    elif self.config.integration == "hypernet":
        flat_weights = self.hypernet(belief)
        logits = self.hypernet.apply_target(flat_weights, obs)
    return sample_from_logits(logits, key), new_state
```

RL²'s use is identical except "belief" is the GRU hidden state rather than a variational posterior.

**Design notes.**
- **Target network must be small.** Target has maybe 100–500 parameters (e.g., 10-dim obs × 16 hidden + biases + 16 × 3 action + biases ≈ 240). A larger target means a larger hypernet output layer, which defeats the purpose: the hypernet's output dim scales *linearly* with target parameter count, so a 10k-param target requires a 10k-output-dim final layer on the hypernet.
- **Separate learning rate.** Hypernet gradients have different magnitudes from those flowing through a standard MLP. Per the implementation-pitfalls rule, the hypernet uses its own `optax` optimizer with `config.hypernet_lr`, distinct from the PPO-core LR used by non-hypernet parameters.
- **JIT-compatibility.** The `apply_target` method uses static shape operations (indexing with known-at-compile-time slices, reshape with static shapes). Everything is `jit`-friendly.
- **No Python-side branching during act.** The `if config.integration == "concat"` branch happens at module construction (once, in `setup`), not inside `act`. Once constructed, the agent either has `self.policy` or `self.hypernet`, and its `act` method jits cleanly.

### `agents/modules/exploration_bonus.py`

The exploration bonus adds an auxiliary intrinsic reward based on belief-space novelty. It's a function that takes a rollout's belief trajectory and returns a per-step bonus to add to the task reward before advantage computation.

**Concrete shape:**

```python
# agents/modules/exploration_bonus.py
import jax.numpy as jnp
from jax import vmap

def compute_exploration_bonus(
    belief_trajectory: jnp.ndarray,    # shape: (T, belief_dim)
    coef: float,
    window_K: int,
) -> jnp.ndarray:
    """L2 distance from rolling mean of last K beliefs, scaled by coef.

    Returns per-step bonus, shape (T,). Bonus at step t uses beliefs from
    steps [max(0, t-K) : t]; at t=0 the bonus is zero.
    """
    T, D = belief_trajectory.shape

    def bonus_at(t):
        # Select the window [t-K, t); pad with current belief so mean is well-defined.
        start = jnp.maximum(0, t - window_K)
        # Use dynamic_slice for jit compatibility:
        window = jax.lax.dynamic_slice(
            belief_trajectory,
            (start, 0),
            (window_K, D),
        )
        # Mask out positions before `start` (when t < K):
        valid = jnp.arange(window_K) < (t - start)
        weights = valid.astype(jnp.float32) / jnp.maximum(valid.sum(), 1)
        window_mean = (window * weights[:, None]).sum(axis=0)
        return coef * jnp.linalg.norm(belief_trajectory[t] - window_mean)

    return vmap(bonus_at)(jnp.arange(T))
```

**How it's integrated.** In `training/rollout.py`, after the rollout is collected, the training loop computes `exploration_bonus` (if `config.exploration_bonus` is True) and adds it to the task reward before GAE:

```python
if config.exploration_bonus:
    bonus = compute_exploration_bonus(
        belief_trajectory=rollout.beliefs,
        coef=config.exploration_bonus_coef,
        window_K=config.exploration_bonus_window,
    )
    augmented_reward = rollout.rewards + bonus
else:
    augmented_reward = rollout.rewards

advantages = compute_gae(augmented_reward, rollout.values, ...)
```

**Design notes.**
- **Belief source is method-specific.** RL² passes its GRU hidden state as `belief_trajectory`; VariBAD passes latent-Gaussian means (μ only, per the mu-only finding from earlier work); Belief-PPO passes the analytical posterior. The bonus function doesn't care about the source — it operates on whatever belief-like vector the agent hands it.
- **Coefficient frozen across methods.** Tuned once in M4; same value used across all factorial cells with `exploration_bonus: true`. Method-specific tuning would conflate "does exploration help?" with "does method X happen to have a favorable bonus coefficient?"
- **Pure JAX, no Python loops.** The `vmap(bonus_at)(arange(T))` pattern computes all bonuses in parallel. No `for t in range(T)` anywhere.
- **No episode-boundary handling here.** The exploration bonus is computed per-episode (each rollout is one episode's beliefs), so window slicing never crosses boundaries. If rollouts contain multiple episodes, the training loop splits them before calling this function.

### `envs/base.py`

```python
class Env:
    def reset(self, key) -> tuple[obs, env_state]: ...
    def step(self, env_state, action, key) -> tuple[obs, env_state, reward, done, info]: ...
```

`env_state` is a pytree containing everything needed for dynamics, *including the latent regime*. This is how `oracles/` and `beliefs/` access ground truth for evaluation while agents only see `obs`.

### Ground-truth leak prevention

`info` dict from `env.step` contains fields for both agent-accessible and evaluation-only information. The training loop splits:
- `agent_info`: what the agent may use during training (e.g. done flags, action masks).
- `eval_info`: what evaluation code sees (true regime, analytical posterior).

Agents must not read `eval_info` except where explicitly allowed (Oracle-PPO reads true regime; Belief-PPO reads analytical posterior; per-regime PPO reads locked regime at env config time, not from step output).

A leak here would silently invalidate results. Enforce with test cases.


---

## Results format

### JSON schema

Each experiment produces one directory under `results/{experiment_name}/` containing:

**`config.json`** — full run config (agent, env, hyperparameters, seed, commit hash).

**`metrics.json`** — per-iteration metrics as arrays (one entry per iteration, typically 100 entries):
```json
{
  "iteration": [0, 1, 2, ..., 99],
  "return_mean": [...],              // mean return across parallel envs
  "return_std": [...],               // std across parallel envs
  "policy_loss": [...],
  "value_loss": [...],
  "entropy": [...],
  "posterior_mse": [...],            // vs analytical HMM posterior, where applicable
  "regime_accuracy": [...],          // where applicable
  "wall_time": [...]                 // cumulative seconds
}
```

Metrics are aggregated *across parallel envs within an iteration*, so each iteration produces one scalar per metric. Confidence intervals in plots come from aggregating across seeds, not across envs within a seed.

**`eval.json`** — final evaluation numbers (return CI, gap closed, per-regime breakdown).

JSON for metrics, not pickle: human-readable, diff-able, stable across Python versions, decouples plotting from training code. Checkpoints (if saved) go in `checkpoints/` subdir as flax serialization.

## Plotting

Plotting scripts take JSON paths as input, produce PNGs as output, have no dependency on training code. Regenerate all thesis figures via `uv run python -m plotting.regenerate_figures`.

This decoupling means:
- Figures can be iterated on during writing without re-running training.
- Re-running plots after config changes is instant.
- Plot code is not in the critical path for training correctness.

All figures committed to `figures/`. Final PNGs directly referenced in LaTeX.

### Learning curve conventions

All learning curves follow the same visual format so figures compare cleanly across milestones:

- **X-axis**: iteration number (0 to ~100). Labelled "Iteration" — not "Environment steps", not "Timesteps". The iteration is the unit of comparison because parallel-env count may differ across experiments.
- **Y-axis**: mean episode return. Labelled with units where meaningful.
- **Line**: mean across seeds, per iteration.
- **Band**: 95% bootstrap CI across seeds (see Statistical methodology section), shaded at alpha=0.2.
- **Reference lines**: where relevant (e.g. Oracle-PPO return, AS analytical optimum), drawn as horizontal dashed lines with labels.
- **Legend**: method names, consistent across figures in a milestone.
- **Color palette**: consistent across all figures in the thesis. Define once in `plotting/style.py`.

### Figure style (committed choices)

These are concrete, committed choices. No fighting matplotlib over them per figure.

**Size.** 5.5 × 3.5 inches for single-column figures; 7.0 × 4.0 for double-column figures that span the LaTeX page width. Bar-chart-with-many-bars figures may go to 7.0 × 3.5 for horizontal breathing room.

**DPI.** 300 for rasterized elements; vector-first where possible (matplotlib's default savefig produces PDF-quality PNG at 300 DPI).

**Font.** Matplotlib default sans-serif at 10pt for axis labels and tick labels, 9pt for legend, 10pt for annotations. Thesis body text is usually 11pt so 10pt in figures reads naturally. Larger than default matplotlib (7-8pt) because 8pt is unreadable at thesis print size.

**No figure titles.** LaTeX captions carry the title. The figure is just axes, labels, legend, and data.

**Gridlines.** Light grey, horizontal only, at major y-axis ticks. Matplotlib `alpha=0.3`. No vertical gridlines (iteration axis doesn't need them). No minor gridlines.

**Spines.** Top and right spines removed (seaborn "despine" style). Left and bottom only.

**Color palette (per-method, committed).** Colorblind-friendly choices, consistent across every plot in the thesis:
- `ppo`: `#7F7F7F` (grey)
- `stacked_ppo`: `#BCBD22` (olive)
- `rl2`: `#1F77B4` (blue)
- `varibad`: `#FF7F0E` (orange)
- `belief_ppo`: `#2CA02C` (green)
- `oracle_ppo`: `#000000` (black)

**Reference lines style.** Ceiling / floor reference lines as horizontal dashed lines, dash pattern `(5, 5)`, line width 1.0, color matching the method (so Oracle-PPO ceiling line is black, Belief-PPO ceiling line is green). Labeled at the right edge of the plot with the method name in the method's color.

**CI bands.** Shaded polygon with `alpha=0.2`, color matching the line. Mean as a solid line at `alpha=1.0`, line width 1.5.

**Error bars (bar charts).** Black, `capsize=3`, `linewidth=1.0`. Always seed-level CI, not within-seed std.

**Legend.** Inside the axes when space permits, outside (right side) otherwise. No frame (`frameon=False`). Font size 9pt.

**File naming.** Snake case, project-prefixed: `fig_rq{n}_*.png` for thesis figures, `fig_M{n}_*.png` for milestone-internal figures. Naming is stable so LaTeX references never break.

All of the above lives in `plotting/style.py` as a function `apply_style()` that's called at the top of every plotting script. No per-script customization; fight style choices once, not per-plot.


---

## Implementation pitfalls to avoid

Meta-RL and PPO have several well-known subtle bugs that silently produce wrong results. The spec prevents each of these by design; this section names them so anyone reviewing the code can verify prevention.

**Shared optimizer across ELBO and PPO losses (VariBAD).** The reconstruction loss, KL term, and PPO loss have different natural learning rates. A single optimizer averages these, training the encoder at a rate optimized for the policy (or vice versa). Design rule: `varibad.py` uses two `optax` optimizers — one for encoder/decoder, one for policy — updating their respective parameter groups independently. Verified via the training loop's `metrics` dict containing `varibad/encoder_lr` and `varibad/policy_lr` as separate values.

**Minibatch shuffling breaking recurrent temporal consistency.** A vanilla PPO update shuffles transitions across timesteps within a minibatch. For recurrent agents, this breaks the temporal order the recurrent state depends on. Design rule: in `training/ppo_update.py`, recurrent agents use trajectory-level minibatches (no within-trajectory shuffling). Agents with `is_recurrent=True` in their metadata trigger this code path automatically.

**Hidden state not reset at episode boundaries.** If the GRU hidden state carries over from one episode to the next, the agent effectively never resets its belief between "tasks," and published meta-RL behavior won't reproduce. Design rule: the training loop detects `done=True` and calls `reset_hidden_for_done` before the next `act` call. Verified by `tests/test_envs.py` (checks done flags trigger reset) combined with M4 validation on bandit (if hidden state leaked across episodes, RL² would be flat; M4 catches this).

**Learning rate shared between hypernet and base policy.** When using hypernet integration, the hypernet and the base policy have different gradient scales; sharing an LR means one is always training too fast or too slow. Design rule: `agents/modules/hypernet.py` takes its own LR from config (default tuned in M4); the base policy uses the PPO core LR from `base_ppo.yaml`.

**VariBAD encoder cold-start.** The encoder's prior at step 0 is untrained; early rollouts feed garbage-belief into the policy. Design rule: the encoder is pretrained on random-policy rollouts for a small warmup phase before joint training begins (warmup length in config, default 10 iterations). If this is skipped, early training is noisy but not catastrophic — this is an optimization not a correctness issue.

**Ground-truth regime leaking into non-Oracle observations.** Cosmetic field names can mislead. If the env's observation dict includes `regime` by accident, any agent that indexes into the obs dict picks it up. Design rule: `beliefs/oracle.py` is the single module that may extract the true regime from env state; only Oracle-PPO, Belief-PPO, and per-regime PPO may import it. Verified by `tests/test_leak.py`.

**Advantages computed across episode boundaries.** GAE with an episode that spans a reset produces wrong advantages. Design rule: advantages reset at episode boundaries; `training/ppo_update.py` masks `done` transitions when computing GAE. Checked by `tests/test_envs.py` which verifies advantage values are identical for standalone-episode rollouts vs multi-episode rollouts up to the first `done`.

**Value-network normalization drift.** Running reward/return normalization statistics computed during training can drift and produce misleading value estimates across seeds. Design rule: normalization statistics are per-seed (not shared across seeds in the same experiment), recomputed from scratch each run, and frozen before evaluation.

**Silent JIT recompilation.** Every shape-drifted call to a `jit`ed function triggers a recompilation; ten of these can double wall-clock time without any visible error. Design rule: all functions exposed to the rollout + update loop have fixed-shape inputs for an entire seed, enforced by the `JAX performance discipline` section. Verified via `stats_M{n}.timing.compile_ratio`, which should stay under 10% on full runs.

**Python control flow inside JIT.** An `if x > 0:` inside a `jit`ed function is evaluated at *trace time* on an abstract array and silently produces wrong behavior (it takes one branch always). Design rule: use `jax.lax.cond`, `jax.lax.select`, `jax.lax.switch`, or masking. No Python `if/else` on array values inside `jit`.

**Non-reproducible randomness.** Creating `PRNGKey(seed)` inside a `jit`'d function bakes the seed in at compile time, so every call uses the same "random" values. Design rule: `PRNGKey` is called exactly once per seed in the outer loop; all subsequent randomness uses `jax.random.split` on keys threaded through function calls.

## Correctness gates (where each one is enforced)

Single-line reference to where each correctness property of the project is actually enforced. Nothing lives only in this section; each item points to the load-bearing implementation.

- **R1 (policy divergence)** — enforced in M2 via `stats_M2_requirements.R1_policy_disagreement.pass`.
- **R2 (locked-regime optimality)** — enforced in M2 diagnostic pass plus full-budget re-verification in M1 and M3.
- **R3 (mixed-regime suboptimality)** — enforced in M2 via `R3_mixed_gap.pass` and re-established in M3 reference levels.
- **R4 (regime inferability)** — enforced in M2 via `R4_inferability.pass`.
- **HMM posterior correctness** — enforced by `tests/test_beliefs.py` (brute-force marginalization on short sequences).
- **No ground-truth leakage** — enforced by `tests/test_leak.py` (regression test on observation dicts for non-oracle agents).
- **Method behavioral validation** — enforced by M4 (published-ordering comparison on bandit + gridworld).
- **Reference-level ordering** — enforced in M3 via `stats_M3_reference_levels.ordering_valid`.

If any gate above is failing, downstream results are not trustworthy. The milestone gate discipline ensures this: passing M2 requires R1–R4; passing M3 requires the ordering; passing M4 requires method validation; passing tests is required for every milestone via `scripts/make_milestone.py`.


---

