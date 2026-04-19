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

Never read more than 2–3 docs in a single session unless the task genuinely spans all of them.

## One-screen project overview

**Problem.** Regime-switching market making is a POMDP where a latent HMM regime governs fill dynamics. Regime-agnostic PPO settles on a compromise policy; belief-conditioned methods can (in principle) do better. The thesis asks how much they actually close the gap, how they decompose the gap, and how their performance scales with difficulty.

**Three research questions (details in `docs/research-questions.md`):**
- **RQ1** — How does the optimality gap decompose into shared-network / inference / compromise-policy costs?
- **RQ2** — How much of each gap component do meta-RL methods close?
- **RQ3** — How does method performance scale with problem difficulty, and does posterior approximation quality predict task performance?

**Method ladder (7 methods, each architecturally distinct):**
1. Regime-agnostic PPO (floor)
2. Stacked-obs PPO (= PPO + `stack_obs` wrapper)
3. RL²
4. VariBAD
5. Belief-PPO (analytical HMM posterior, ceiling)
6. Oracle-PPO (true regime, ceiling)
7. Per-regime PPO (ceiling)

Plus two orthogonal ablation axes composed via config flags, not separate agent files:
- **Integration** — concat vs hypernet
- **Exploration bonus** — off vs on (L2 novelty on whatever representation the method maintains)

Full ladder + ablation spec in `docs/research-questions.md` → "Method ladder".

## Reference levels (quick reference)

Four reference performance levels, established in M3 (see `docs/milestones/m3.md`):

| Level | Role | Regime info |
|---|---|---|
| **Per-regime PPO** | ceiling | trained on locked regime, reports conditional optimum |
| **Oracle-PPO** | ceiling | conditioned on true regime one-hot |
| **Belief-PPO** | ceiling | conditioned on analytical HMM posterior |
| **Regime-agnostic PPO** | floor | no regime information |

Expected ordering: `regime_agnostic ≤ belief ≤ oracle ≤ per_regime`.

The three *ceilings* (per-regime, oracle, belief) minus the *floor* (regime-agnostic) gives the total optimality gap that meta-RL methods are asked to close.

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
- [ ] M2 — regime-switching env, R1–R4 verified
- [ ] M3 — reference levels (answers RQ1)
- [ ] M4 — implementation validation
- [ ] M5 — ladder + factorial on MM (answers RQ2)
- [ ] M6 — difficulty sweep + decoupling (answers RQ3)
- [ ] M7 — supplementary ablations (optional)

Current env version: `E_final = ???` (symlink to the env config that passed R1–R4 in M2).

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

