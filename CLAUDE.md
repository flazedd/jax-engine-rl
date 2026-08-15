# CLAUDE.md

Project: master's thesis on **belief-conditioned meta-RL for regime-switching market making**, supervised at TU Delft by Frans Oliehoek and Fenghui.

This file is the navigation hub. Full specifications live in `docs/`. **Load only the docs relevant to the current task** — reading all of them at once defeats the purpose of splitting.

## How to use this file

1. Read this file every session (it's small on purpose).
2. Decide what you're working on, pick the relevant docs from the table below, and read those.
3. When running or changing the experimental programme, read `docs/pipeline.md` first.

## Where to look

| For work on... | Read |
|---|---|
| Running the programme, the results layout, provenance, the dummy layer | `docs/pipeline.md` |
| Understanding research questions, plot descriptions, what the thesis claims | `docs/research-questions.md` |
| Statistical protocol, comparison sets, pre-registered hypotheses | `docs/methodology.md` + `evaluation/protocol.py` |
| Environment design (R1–R4), env-iteration process, action space | `docs/environment.md` |
| Writing agent code, JSON result schemas, plotting modules, known pitfalls | `docs/implementation.md` |
| uv, YAML configs with `extends:`, tests, run modes, JAX performance rules | `docs/conventions.md` |
| Writing or editing any figure — naming, legends, axes, compute footer | `docs/plotting.md` |
| Why something is the way it is (M0–M7 history, rejected env designs) | `docs/_archive/` |

Never read more than 2–3 docs in a single session unless the task genuinely spans all of them.

## One-screen project overview

**Problem.** Regime-switching market making is a POMDP where a latent HMM regime governs fill dynamics. Regime-agnostic PPO settles on a compromise policy; belief-conditioned methods can (in principle) do better. The thesis asks how much they actually close the gap, how they decompose the gap, and how their performance scales with difficulty.

**Three research questions (details in `docs/research-questions.md`):**
- **RQ1** — How do RL² and VariBAD perform under each conditioning architecture, concatenation and hypernetwork?
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

Three reference performance levels (§3.4 of the thesis):

| Level | Role | Regime info |
|---|---|---|
| **Oracle-PPO** | ceiling | conditioned on true regime one-hot |
| **Belief-PPO** | ceiling | conditioned on analytical HMM posterior |
| **Regime-agnostic PPO** | floor | no regime information |

Expected ordering: `regime_agnostic ≤ belief ≤ oracle`.

The reference gap `oracle − regime_agnostic` decomposes into the **posterior-information component** (`oracle − belief`) and the **belief-to-policy component** (`belief − agnostic`). Both are estimated under the matched protocol, not theoretical bounds.

## Critical conventions (applied every turn)

These are the non-negotiable rules. Full discussion in `docs/conventions.md`.

- **Use uv for every command.** `uv run python -m module.path` — never plain `python`. See `docs/conventions.md` → "Running scripts with uv".
- **No shell scripts.** All orchestration is Python, invoked via `uv run`. The only shell is `git clone && cd && uv sync` in README.
- **JSON is for decisions, PNG is for the human.** Every pass criterion is a JSON field with a threshold. Claude Code reads JSON, never PNGs.
- **Every script writes a shared-schema summary JSON** and prints a single final line `[script] OK | key=val | output=path` (or `FAIL | reason=...`).
- **YAML configs use `extends:` composition.** Base configs hold shared hyperparameters; experiment configs override minimally. See `docs/conventions.md` → "Configuration files (YAML)".
- **Tune-once-freeze hyperparameters.** Optimiser settings are defaults and are never varied by method or environment; the entropy coefficient is the one documented exception (§3.8). Never retune per experiment.
- **Seed discipline.** 20 seeds, numbered 0-19, shared across methods so every comparison is paired. The count lives in `evaluation/protocol.py`, not in prose. Never delete seeds silently.
- **Three comparison sets**, corrected separately by Holm-Bonferroni: returns, behavioural diagnostics, belief quality. Sizes in `evaluation/protocol.py`. Everything else is descriptive.
- **The dummy chain must be green before a real run.** `--dummy` proves the analysis and figure chain in seconds; preflight enforces it.
- **JAX performance is not optional.** Shape-stable inputs, `lax.scan` / `vmap`, no Python loops in jit regions. See `docs/conventions.md` → "JAX performance discipline".

## Current state

The repository is organised by the structure of the thesis, not by the order
the work happened. `docs/pipeline.md` describes the layout, the programme
phases and the provenance rules; `evaluation/protocol.py` holds the statistical
constants that Table 3.5 and Table D.8 report.

One command reproduces everything the thesis reports:

```
uv run python -m scripts.run_matched_programme            # full programme
uv run python -m scripts.run_matched_programme --dummy    # whole chain on synthetic data, ~3s
uv run python -m scripts.status                           # where a running programme is
uv run python -m scripts.thesis_contract --strict         # artifacts against the protocol
```

The M0–M7 milestone log, including the environment-design iterations and the
reasons several designs were rejected, is in `docs/_archive/milestone-history.md`.

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
- **M0–M7** — the project's original progress milestones, retired. See `docs/_archive/`.
- **RQ1, RQ2, RQ3** — the three research questions. See `docs/research-questions.md`.

