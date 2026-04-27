# CLAUDE.md

Project: master's thesis on **belief-conditioned meta-RL for regime-switching market making**, supervised at TU Delft by Frans Oliehoek and Fenghui.

This file is the navigation hub. Full specifications live in `docs/`. **Load only the docs relevant to the current task** — reading all of them at once defeats the purpose of splitting.

## How to use this file

1. Read this file every session (it's small on purpose).
2. Decide what you're working on, pick the relevant docs from the table below, and read those.
3. When executing a specific milestone, read only that milestone's file from `docs/milestones/`, plus `contingency.md`.

## Where to look

| For work on... | Read |
|---|---|
| Understanding research questions, plot descriptions, what the thesis claims | `docs/research-questions.md` |
| Running or debugging a specific milestone | `docs/milestones/m{n}.md` + `docs/milestones/contingency.md` |
| Cross-milestone planning, the dependency graph, "when is the thesis done?" | `docs/milestones/overview.md` |
| Statistical tests, pre-registered hypotheses, limitations, posterior-quality probe | `docs/methodology.md` |
| Environment design (R1–R4), env-iteration process, action space justification | `docs/environment.md` |
| Writing agent code, implementing hypernet / exploration bonus, JSON result schemas, plotting modules, known pitfalls | `docs/implementation.md` |
| Setting up uv, YAML configs with `extends:`, git workflow, tests, run modes, JAX performance rules, script output format, hyperparameter discipline | `docs/conventions.md` |
| Writing or editing any milestone figure — naming, legends, y-axis, bar labels, compute-budget footer, reference lines | `docs/plotting.md` |

Never read more than 2–3 docs in a single session unless the task genuinely spans all of them.

## One-screen project overview

**Problem.** Regime-switching market making is a POMDP where a latent HMM regime governs fill dynamics. Regime-agnostic PPO settles on a compromise policy; belief-conditioned methods can (in principle) do better. The thesis asks how much they actually close the gap, how they decompose the gap, and how their performance scales with difficulty.

**Three research questions (details in `docs/research-questions.md`):**
- **RQ1** — How does the optimality gap decompose into shared-network / inference / compromise-policy costs?
- **RQ2** — How much of each gap component do meta-RL methods close?
- **RQ3** — How does method performance scale with problem difficulty, and does posterior approximation quality predict task performance?

**Method ladder (6 methods, each architecturally distinct):**
1. Regime-agnostic PPO (floor)
2. Stacked-obs PPO (= PPO + `stack_obs` wrapper)
3. RL²
4. VariBAD
5. Belief-PPO (analytical HMM posterior, ceiling)
6. Oracle-PPO (true regime, ceiling)

Plus two orthogonal ablation axes composed via config flags, not separate agent files:
- **Integration** — concat vs hypernet
- **Exploration bonus** — off vs on (L2 novelty on whatever representation the method maintains)

Full ladder + ablation spec in `docs/research-questions.md` → "Method ladder".

## Reference levels (quick reference)

Three reference performance levels, established in M3 (see `docs/milestones/m3.md`):

| Level | Role | Regime info |
|---|---|---|
| **Oracle-PPO** | ceiling | conditioned on true regime one-hot |
| **Belief-PPO** | ceiling | conditioned on analytical HMM posterior |
| **Regime-agnostic PPO** | floor | no regime information |

Expected ordering: `regime_agnostic ≤ belief ≤ oracle`.

Total optimality gap = `oracle − regime_agnostic`, decomposed 2-way into **inference cost** (`oracle − belief`) and **compromise-policy cost** (`belief − agnostic`). Meta-RL methods in M5 are evaluated against these components.

## Critical conventions (applied every turn)

These are the non-negotiable rules. Full discussion in `docs/conventions.md`.

- **Use uv for every command.** `uv run python -m module.path` — never plain `python`. See `docs/conventions.md` → "Running scripts with uv".
- **No shell scripts.** All orchestration is Python, invoked via `uv run`. The only shell is `git clone && cd && uv sync` in README.
- **JSON is for decisions, PNG is for the human.** Every pass criterion is a JSON field with a threshold. Claude Code reads JSON, never PNGs.
- **Every script writes a shared-schema summary JSON** and prints a single final line `[script] OK | key=val | output=path` (or `FAIL | reason=...`).
- **YAML configs use `extends:` composition.** Base configs hold shared hyperparameters; experiment configs override minimally. See `docs/conventions.md` → "Configuration files (YAML)".
- **Tune-once-freeze hyperparameters.** PPO core tuned in M1 then frozen project-wide. Method-specifics tuned in M4 then frozen. Never retune per experiment.
- **Seed discipline.** 5 seeds for primary experiments, fixed values `{0, 1, 2, 3, 4}` across methods to enable paired statistical tests. Never delete seeds silently.
- **Pre-registered primary hypotheses.** 6 for RQ2, 3 for RQ3. Everything else is exploratory. See `docs/methodology.md`.
- **Git tag every passing milestone** (`git tag m{n}-passed`). Main branch always passes the latest milestone.
- **JAX performance is not optional.** Shape-stable inputs, `lax.scan` / `vmap`, no Python loops in jit regions. See `docs/conventions.md` → "JAX performance discipline".

## Current status

(Update this section as milestones pass.)

- [x] M0 — infrastructure skeleton
- [x] M1 — PPO on AS baseline
- [x] M2 — regime-switching env, R1–R4 verified on E6e_symmetric_kappa05 (current E_final; E2/E3 also passed)
- [x] M3 — reference levels on E6e (agnostic=136.2, belief=168.5, oracle=180.1; gap=43.9, compromise=32.3 / inference=11.6)
- [x] M4 — implementation validation (RL² & VariBAD clear PPO floor on bandit/gridworld/regime_bandit at full budget; see `results/milestones/M4/method_ranking.json`)
- [ ] M5 — ladder + factorial on MM (answers RQ2). Stage A failed initial pass (2026-04-25); recovery via Steps 1-5. **Step-4 full-budget 4-cell factorial complete (n=8 × 200 iter, 2026-04-27)**: RL² hypernet **168.56** / VariBAD hypernet **166.80** (both at Belief-PPO ceiling 168.5, +30-32 above floor 136.2, gap_closed_vs_belief 0.95-1.00); RL² concat 124.07 / VariBAD concat 110.32 (both below floor). Both primary hypotheses (`hypernet_beats_concat`) supported with Holm-corrected p=0.0156, |Δ|≈45-55, robust to LOO. **Posterior probe**: all 4 cells decode regime at 0.79-0.81 vs analytical 0.88 — concat and hypernet have equivalent belief decodability, integration is the bottleneck (decoupling finding). `ranking_stable: false` (5/8 seeds — instability between the two hypernet cells which are tied within seed noise); decisive scientific claims robust. Tag `m5-passed` held for manual review. See `STEP4_FINDINGS.md`.
- [ ] M6 — difficulty sweep + decoupling (answers RQ3)
- [ ] M7 — supplementary ablations (optional)

Current env version: `E_final = e6e_symmetric_kappa05` (symlink at `experiments/configs/envs/e_final.yaml`). E6e is symmetric within-regime fills (bid-side prob == ask-side prob) so regime no longer drives directional inventory drift, paired with κ=0.05 inventory penalty. Three regimes vary the per-regime (p_tight, p_wide) magnitudes so the optimal *action class* (sym vs favor_X) differs across regimes — but inventory does not leak regime info to a memoryless policy. M2 verify (40 iter × 256 envs × 3 seeds): agnostic=101.7, belief=127.3, oracle=141.1 → total gap 39.4 (compromise-policy cost 25.6, inference cost 13.8). Mid-mode (100 iter × 1 seed) gap was 47 vs E3's ~0. All R1–R4 pass. Replaces E3 because E3's regime-driven directional fills made inventory a near-sufficient statistic for regime, collapsing the empirical compromise gap to ~0 even though analytical compromise_VI predicted 21.

## Glossary

Defined once here, used freely across all docs. Lives in `CLAUDE.md` (not a separate file) because it's referenced constantly.

**Reinforcement learning / meta-RL**
- **RL** — reinforcement learning.
- **MDP** — Markov decision process.
- **POMDP** — partially observable Markov decision process.
- **PPO** — Proximal Policy Optimization (Schulman et al., 2017).
- **SAC** — Soft Actor-Critic.
- **VI** — value iteration.
- **MLP** — multilayer perceptron (feedforward network).
- **GRU / LSTM** — gated recurrent unit / long short-term memory (recurrent network types).
- **GAE** — generalized advantage estimation.
- **RL²** — recurrent meta-RL method; policy is a recurrent network that treats the episode as inner-loop learning (Duan et al., 2016).
- **VariBAD** — variational Bayes-adaptive deep RL; meta-RL with an explicit variational posterior over tasks (Zintgraf et al., 2020).
- **MAML** — Model-Agnostic Meta-Learning. Excluded from this thesis; mentioned only for scope boundaries.
- **HN** — hypernetwork; a network whose output is the parameters of another network.

**Probability / information theory**
- **HMM** — hidden Markov model.
- **KL** — Kullback-Leibler divergence.
- **ELBO** — evidence lower bound (variational inference objective).
- **MSE** — mean squared error.
- **CI** — confidence interval. Throughout this project, "CI" refers specifically to a **95% bootstrap confidence interval computed across training seeds** (not across parallel envs within a seed). Bootstrap rather than parametric because seed counts are low (n=5 default) and the t-distribution assumption is unsafe. Computed via 10,000 bootstrap resamples of the per-seed final-return values. Plotted as a shaded band around the mean line. Full details in `docs/methodology.md`.

**Finance / market making**
- **MM** — market making.
- **AS** — Avellaneda-Stoikov, the standard reference MM model (Avellaneda & Stoikov, 2008).
- **LOB** — limit order book.

**Project-internal**
- **R1–R4** — the four problem requirements (policy divergence, locked-regime optimality, mixed-regime suboptimality, regime inferability) that any valid env parameterization must satisfy. See `docs/environment.md`.
- **E0, E1, E2, ...** — iterations of the env design process, starting from vanilla AS (E0) and adding structural elements one at a time.
- **M0–M7** — the project's progress milestones. See `docs/milestones/`.
- **RQ1, RQ2, RQ3** — the three research questions. See `docs/research-questions.md`.

