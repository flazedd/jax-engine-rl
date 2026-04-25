# Conventions and operational discipline

All the "how we do things" rules: uv, YAML configs, tests, git, reproducibility, milestone discipline, compute budgets, run modes, JAX performance, script output format, hyperparameter freeze discipline.

Read this when setting up the project, writing a new config, adding a test, committing work, or touching anything performance-sensitive.

See also:
- `docs/implementation.md` — module-level rules (interface contracts, pitfalls).
- `docs/methodology.md` — statistical rules (separate from operational conventions).

---

## Compute plan

### Hardware

Mac mini M4 CPU. JAX CPU backend. No GPU.

### Training loop structure

Training is structured as a small number of **iterations** (outer loop steps), each running many **parallel environments** in a batched rollout. This trades total environment steps for statistical stability within each iteration — short curves, low noise.

**Core shape (non-negotiable):**
- **Iterations (outer loop)**: capped at 100. This is the x-axis on learning curves.
- **Parallel envs per iteration**: large (hundreds to low thousands), `vmap`-batched. Exact count tuned per experiment to fit M4 memory.
- **Rollout length per env per iteration**: short horizon (e.g., 128–512 steps per env per iteration).
- **Total env interactions per iteration** = parallel_envs × rollout_length. This is what the PPO update sees.

**Why this shape.** A learning curve with 100 iterations and many parallel envs per iteration is a **short line with low variance per point**. Each iteration aggregates across hundreds of parallel envs, giving a stable per-iteration estimate of return. 100 iterations is enough to see convergence but few enough to keep wall-clock manageable. Keep iterations low and parallel envs high — not the reverse.

**Statistical stability discipline.** If per-iteration returns are noisy (jagged learning curve even after seed averaging), the fix is **more parallel envs**, not more iterations. Variance across envs within an iteration should be reported in metrics.json so you can diagnose whether a jagged curve is genuinely high gradient variance or just too few envs averaged.

**JIT amortization.** The compiled rollout + update step is called exactly 100 times per seed. Keep function signatures shape-stable (fixed parallel_envs, fixed rollout_length, fixed batch shapes) so there's one compilation per seed, not recompilations mid-run.

**Seeds.** 8 seeds per method for primary experiments (M5/M6); 5 for M3 reference levels; 3 for validation/diagnostic milestones (M2/M4). Run sequentially (parallelism is inside each seed via `vmap` over envs, not across seeds). Seed-level confidence intervals reported. Pre-registered in the config. See `methodology.md` → "Seed protocol" for the n=8 derivation.

### Implications

- JIT compilation overhead is one-time per seed — compile the full rollout + update step once, then run 100 iterations. Keep function signatures shape-stable.
- Env throughput matters less than at long-horizon training; agent forward/backward pass dominates.
- Ladder scope pre-commitment remains important. Even with short curves, 7 ladder methods + 6 additional factorial cells + mu-only ablation = 14 unique configurations × 8 seeds on MM is non-trivial wall-clock — plan for overnight runs in M5.

### JAX performance discipline

JAX's performance comes from aggressive compilation, not from being a numpy replacement. The difference between a JAX project running at 10% of hardware and at 90% is almost entirely about whether the code is written to compile well. The rules below are committed design constraints; every module in the project follows them.

**Compile the largest possible function.** The unit of compilation is `jit`ted, and JIT overhead is amortized over how much work it does. Compile the whole rollout + update step for a seed into one function, not one call per env-step. Concretely, `training/train.py` wraps the entire (reset-to-convergence) inner loop in `jit` where possible, or at least the per-iteration (rollout + GAE + update) step. Compile once at the start of a seed, run 100 iterations, never recompile mid-run.

**Shape-stability is non-negotiable.** Every `jit`ted function has fully static shapes in and out. This means:
- `parallel_envs`, `rollout_length`, `batch_size` are known at compile time and fixed for an entire seed.
- No Python-side `if shape == ...` branching inside `jit` regions.
- No dynamic `jnp.arange(n)` where `n` varies across calls. Use `jax.lax.dynamic_slice` when you need indexing with runtime indices.
- Padding to fixed shapes is fine and often necessary. The cost of computing over slightly-more elements is far smaller than the cost of triggering recompilation.

**Use `lax.scan` for sequential structure, not Python loops.** Anywhere you'd be tempted to write `for t in range(T)` inside a JAX function (rollouts, GAE, recurrent state updates, HMM forward algorithm), use `jax.lax.scan` instead. Python loops inside JIT are either unrolled (compile-time blow-up) or cause recompilation per iteration. `scan` is the correct pattern.

**Use `vmap` for batch parallelism, not Python loops or list comprehensions.** Parallel envs are batched via `vmap`, not via a Python loop over env instances. Every env interaction — reset, step, reward, done detection — is written as a scalar function and vectorized with `vmap`. Same rule for computing advantages across envs, or per-seed bootstrap resampling.

**Minimize pytree structure churn.** Every `act`/`update` call takes and returns pytrees with *identical structure* across calls. Adding or removing keys mid-run invalidates the compilation cache. The `AgentState` pytree schema is fixed for a seed; if metadata varies, put it in a sidecar that's not part of the JIT'd function.

**Random keys are threaded, not created ad hoc.** `jax.random.PRNGKey(seed)` is called once per seed at the start; all subsequent randomness is produced by `jax.random.split`. Never call `PRNGKey` inside a JIT'd function — the seed would be baked in at compile time, so every call uses the same "random" values.

**Pure functions only inside JIT.** No side effects, no logging, no file I/O, no Python-side mutation of state. Side effects go in the outer loop that calls the JIT'd function. If you need to log per-iteration metrics, the JIT'd function *returns* metrics as part of its output pytree, and the outer loop writes them out after the call completes.

**Static shapes for env state.** `env_state` pytrees are shape-stable across the entire rollout, including after `done=True`. An env with variable-length episodes still uses a fixed-shape `env_state` plus a `done` mask. The rollout never returns early — it scans for `rollout_length` steps every time, and `done` flags are used in GAE to mask advantages at episode boundaries.

**Avoid `jax.device_put` inside loops.** Host-to-device transfers are expensive on any backend. Data lives on device from the moment it's created until results are read out.

**Profile if in doubt.** `jax.profiler.trace()` gives a flame-graph showing compile time vs execution time per function. If a milestone is slower than expected, profile before guessing. Symptoms of bad compilation: first iteration takes forever (expected), *second* iteration also takes forever (bad — probably recompiling due to shape drift).

**Specific idiom checklist.** For every new function in the project, check:
- [ ] Marked with `@jax.jit` or called inside a `jit`'d caller?
- [ ] All inputs have fixed shapes across calls?
- [ ] No Python control flow depending on array values (use `lax.cond` / `lax.switch` if needed)?
- [ ] No `for` loops that should be `scan` or `vmap`?
- [ ] Returns pytrees with fixed structure?
- [ ] Takes a `key` parameter for any randomness, doesn't create its own?

**Common footguns and their fixes.**
- **Recompilation every iteration**: shape of some input is drifting. Log shapes at function entry; one of them is lying about being static.
- **Slow first iteration, slow again later at iteration ~N**: you're hitting the JIT cache limit. Consolidate compiled functions.
- **NaN gradients that appear only at certain batch sizes**: numerical instability in softmax or log computations. Use `jax.nn.log_softmax` and `jax.nn.logsumexp` rather than manual `log(softmax(...))`.
- **`vmap` over an env with state-dependent branching fails**: env `step` must be written with `lax.cond`/`lax.select` rather than `if/else`. Every branch of a conditional runs; `lax.select` picks the output.
- **Rollout loop is Python-slow despite JIT**: the `scan` body is reconstructing Python objects. Ensure the body returns raw arrays, not custom classes, and that the carry is a flat pytree.

**JAX performance is verified, not assumed.** Each milestone's build order ends with a run-mode test that records iteration time. Track this in `stats_M{n}_*.json`:
```json
"timing": {
  "compile_time_seconds": ...,    // time for first iteration
  "per_iter_time_seconds": ...,   // median over iterations 2–N
  "iterations_run": ...,
  "compile_ratio": ...            // compile / (compile + total_run)
}
```

If `per_iter_time` drifts upward or `compile_ratio` is suspiciously high (>10% on full runs), something is recompiling. This is a diagnostic, not a pass criterion — but if it's degenerate, investigate before continuing.

### Configuration

All iteration counts, parallel env counts, rollout lengths, seed counts, and hyperparameters live in **one file per experiment** under `experiments/configs/`. No magic constants in code. The config file is the single source of truth for a run and is copied into `results/{experiment_name}/config.json` at run start for reproducibility.

### Hyperparameter discipline

The default failure mode for RL projects is per-experiment hyperparameter tuning that doubles or triples the true compute cost and creates irreproducible numbers. Avoid this with the following discipline:

**Tune once, freeze.** PPO core hyperparameters (learning rate, clip ratio, entropy coefficient, GAE lambda, value loss coefficient, minibatch size, epochs per update) are tuned *once* on the AS baseline in M1. From that point on, these values are frozen for the entire project. Every method that uses PPO internally — Oracle-PPO, Belief-PPO, per-regime PPO, the PPO inside RL² and VariBAD — uses the same core values.

**Method-specific hyperparameters are tuned during M4 validation** and then frozen for M5/M6:
- VariBAD: KL weight, latent dim, encoder/decoder sizes, reconstruction loss weight.
- RL²: recurrent hidden dim, input augmentation choice ((obs, prev_action, prev_reward) by default).
- Hypernet integration: hypernet hidden dim, target layer selection.
- Exploration bonus: bonus coefficient, novelty window size.

Each tuned per method on the cheapest validation task where the effect is visible, documented in the method's config file with a comment noting where the value came from.

**No retuning per experiment.** If a method wants different hyperparameters on MM than on the bandit, that's a *finding* (and a concerning one — it suggests hyperparameter sensitivity), not a convenience. Report it; do not silently retune.

**Sweeps are pre-registered.** If a hyperparameter sweep is done, it goes in the config with explicit values tried and the selection criterion. No post-hoc picking of the best seed.

**Composable configs.** Configs are built from a shared base (e.g., `base_ppo.yaml` has the core PPO values) and experiment-specific overrides (e.g., `m5_varibad.yaml` extends the base and adds VariBAD-specific values). This prevents drift where "PPO learning rate" accidentally has a different value in different experiments.

### Run modes (`--super-fast`, `--fast`, full)

Every script that can take longer than ~30 seconds supports three run modes via CLI flags. The flag modifies the loaded config in memory before execution — it never requires a separate config file.

**Both `--super-fast` and `--fast` use 1 seed.** Seed-level statistics are only meaningful in full runs. Fast modes are for pipeline verification and single-seed directional signal, not for claims about variance or ranking.

**`--super-fast`** — smoke test. Purpose: verify the pipeline compiles and runs end-to-end. Target runtime: under 30 seconds.
- Iterations: 2 (just enough to confirm the outer loop executes).
- Parallel envs: small (e.g., 16).
- Rollout length: short (e.g., 32).
- Seeds: **1**.
- JIT compilation still happens (this catches shape bugs and import errors).
- Output artifacts: JSON produced, plot optionally skipped or rendered as a placeholder. JSON must still validate against the expected schema — this is the primary check.

**`--fast`** — smoke test with signal. Purpose: verify that *learning is happening*, not just that the code runs. Target runtime: a few minutes.
- Iterations: 20.
- Parallel envs: moderate (e.g., 64–128).
- Rollout length: moderate (e.g., 128).
- Seeds: **1**.
- Output artifacts: full JSON and plot. Learning curve should show an upward trend even if not converged. Final return won't match full-run numbers and milestone pass criteria do not apply, but the slope of the curve tells you the method is functional.

**No flag (full)** — production run. Purpose: generate milestone/thesis artifacts.
- Iterations: 100.
- Parallel envs: config default (tuned per experiment to fit M4 memory).
- Rollout length: config default.
- Seeds: 5 (minimum).
- All milestone pass criteria apply.

**Implementation.** The flag is handled centrally in `training/config.py`:
```python
def apply_run_mode(cfg, mode):
    if mode == "super_fast":
        cfg.iterations = 2
        cfg.parallel_envs = 16
        cfg.rollout_length = 32
        cfg.num_seeds = 1
    elif mode == "fast":
        cfg.iterations = 20
        cfg.parallel_envs = 128
        cfg.rollout_length = 128
        cfg.num_seeds = 1
    return cfg
```

Every experiment script imports and calls this. No script implements its own version.

**When to use which.**
- `--super-fast` on every commit / before every long run / after every refactor. Five seconds of wait is cheap insurance against a crash at hour three.
- `--fast` when tweaking hyperparameters or debugging a method that isn't converging. Fast enough to iterate, slow enough to see whether the fix helped. Single seed — if behaviour seems seed-dependent, confirm with a full run rather than interpreting fast-mode noise.
- No flag for anything that produces a milestone artifact or a thesis figure.

**Plot conventions for fast modes.** Learning curves from fast modes are rendered without CI bands (single seed has no CI). Plotting code should detect `num_seeds == 1` from the summary JSON and draw a single line rather than a band — so the plot still looks sensible and isn't mistaken for a full run.

### Script output discipline

Every script that runs — training, evaluation, plotting, tests, utilities — produces **discrete, verifiable output** that confirms whether it worked.

**JSON is for decisions, PNG is for the human.** This is a load-bearing rule:

- **Claude Code reads JSON.** Every numeric or boolean pass criterion must be a field in the script's output JSON. Automated verification, milestone pass/fail decisions, and tuning loops all operate on JSON.
- **Claude Code does not read PNGs.** PNGs exist for the human (thesis reader, supervisor, you) to verify things make sense visually. They are not input to automated decisions.
- **If a pass criterion currently reads like "curve should look X,"** it must be operationalized as a JSON field: a ratio, a boolean, a fitted slope, a computed entropy decay. Visual judgments are not pass criteria for Claude Code.
- **When Claude Code finds itself wanting to inspect a PNG to make a decision, the JSON schema is incomplete.** Add the missing field to the JSON instead.

This is also a token-cost rule — reading a PNG costs ~2000 tokens; reading the equivalent JSON costs ~100. Over a long agentic session, the difference compounds significantly.

**Required output from every script.**
1. **A summary JSON** written to a predictable path. Contains enough stats that Claude Code can make every downstream decision without touching a PNG.
2. **A final line to stdout** of the form:
   ```
   [SCRIPT_NAME] OK | key_stat_1=... | key_stat_2=... | output=path/to/artifact
   ```
   or on failure:
   ```
   [SCRIPT_NAME] FAIL | reason=...
   ```
   This line is greppable, parseable, and fits on one terminal line. Long runs can also print intermediate progress lines — but the final line is what makes the run verifiable without reading the full log.

**Exit code.** Non-zero on any unexpected failure. Zero only if the script completed and the output JSON was written.

**No silent success.** A script that completes without producing its expected JSON is a bug, even if it didn't crash. The output JSON is the contract.

**Examples of the output discipline in action.**

*Training script (`training/train.py`):*
- Writes `results/{experiment}/metrics.json` and `results/{experiment}/eval.json`.
- Final stdout: `[train] OK | method=varibad | final_return=12.4 | gap_closed=0.67 | output=results/varibad_mm/`

*Plotting script (`plotting/learning_curves.py`):*
- Writes a summary JSON noting which input files were read and what was plotted.
- Final stdout: `[plot_learning_curves] OK | input_runs=5 | methods=['ppo','rl2','varibad'] | output=figures/milestones/M5/fig_rq2_learning_curves.png`

*Test script (`tests/test_beliefs.py`):*
- Writes a test report JSON with per-test pass/fail.
- Final stdout: `[test_beliefs] OK | tests_run=12 | tests_passed=12 | output=results/tests/beliefs.json`

*VI oracle (`oracles/value_iteration.py`):*
- Writes convergence stats and final policy.
- Final stdout: `[vi] OK | converged_at_iter=847 | bellman_residual=1.2e-7 | output=results/oracles/vi_regime_0.json`

**Why this matters.** When running dozens of experiments in batches (e.g., `run_ladder.py`), the only practical way to verify everything worked is `grep "OK\|FAIL" logs/*.log`. Without this discipline, failure diagnosis becomes archaeology through full logs.

**Script output schema.**

Every script's summary JSON conforms to a minimal shared schema:
```json
{
  "script": "train",
  "status": "OK",
  "started_at": "2026-...",
  "finished_at": "2026-...",
  "run_mode": "super_fast|fast|full",
  "config_used": { ... the effective config after run-mode overrides ... },
  "key_stats": { ... script-specific stats ... },
  "outputs_written": [ "path/to/artifact1", "path/to/artifact2" ],
  "error": null
}
```

Every script writes this alongside its main output. Enables automated milestone verification and lets `scripts/make_milestone.py` check that every step produced its expected artifacts before regenerating plots.


---

## Conventions

### Language and environment

- Python 3.11+, managed with **uv** (the package manager; see https://docs.astral.sh/uv/). The project uses uv exclusively for dependency management and script execution. No `pip install`, no manual venv activation.
- JAX-native throughout; no PyTorch or NumPy-only agents.
- Type hints on public interfaces.
- **No shell scripts.** All orchestration (milestone runners, ladder batch runners, sweep runners) is Python, invoked via `uv run python -m ...`. Shell scripts bypass uv's locked environment, compose poorly with JSON output discipline, and are harder for Claude Code to debug than Python modules. The only shell commands in the project are the three-line clone-and-sync in README.md.

### Running scripts with uv

**Every Python command in this document uses `uv run`.** Never `python -m ...` directly.

Why: `uv run` guarantees the script executes in the project's locked environment — the exact Python version and dependency versions pinned in `pyproject.toml` and `uv.lock`. Plain `python -m` picks up whatever environment happens to be active, which on a development machine is often wrong and silently produces results that won't reproduce on a fresh clone.

The pattern is:

```
# correct — uses the project's locked environment:
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml

# wrong — depends on shell state:
uv run python -m training.train --config experiments/configs/m1_ppo_as.yaml
```

The commands listed in each milestone assume `uv run` as the prefix. They're shown as `uv run python -m ...` throughout.

**First-time setup:**
```
git clone <repo>
cd thesis
uv sync          # installs all deps from pyproject.toml + uv.lock
# verify:
uv run python -c "import jax; print(jax.devices())"
```

**Adding a dependency** (if ever needed):
```
uv add <package>     # updates pyproject.toml and uv.lock
git commit pyproject.toml uv.lock
```

Do not edit `pyproject.toml` by hand for dependencies; let uv manage it.

### Configuration files (YAML)

Every experiment is defined by a single YAML file under `experiments/configs/`. YAML is the format for two reasons: (a) human-readable and diffable in git (unlike pickle or JSON with embedded code), and (b) composable — base configs can be extended by experiment-specific overrides, preventing drift across runs.

**How YAML files get used.**

1. The user invokes a training or verification script with `--config path/to/config.yaml`.
2. `training/config.py` reads the YAML file, resolves any `extends:` directives (loading the base and deep-merging overrides on top), and materializes a typed `ExperimentConfig` dataclass.
3. If a CLI flag like `--super-fast` or `--fast` is passed, `apply_run_mode(cfg, mode)` mutates the dataclass in memory (overrides iterations, parallel envs, rollout length, seed count).
4. The effective config (after all resolution and run-mode overrides) is serialized to JSON and written to `results/{experiment_name}/config.json` at the start of the run, alongside the git commit hash. This is the reproducibility record — anyone can reproduce a run from `{config.json, commit_hash, seed}`.
5. The script then proceeds with the typed dataclass; no YAML is read after startup. No magic strings at runtime.

**Composition via `extends`.**

A typical experiment config is short — maybe 10 lines — because it extends a shared base:

```yaml
# experiments/configs/m5_ladder_varibad.yaml
extends: base/base_varibad.yaml
experiment_name: m5_ladder_varibad
env:
  extends: envs/e_final.yaml
integration: concat
exploration_bonus: off
iterations: 100
num_seeds: 8
```

The base file `base/base_varibad.yaml` holds the tuned PPO core hyperparameters, the VariBAD-specific parameters (KL weight, latent dim), and defaults for everything else. The experiment file only specifies what's different.

This is what makes the hyperparameter discipline enforceable — if "PPO learning rate" appears in one place (`base_ppo.yaml`), it can't drift across runs. The alternative (one full config per experiment, hand-copied) is where silent parameter drift comes from.

**Multiple extends.** When a config needs to compose two orthogonal bases (e.g., VariBAD agent + hypernet integration), `extends:` accepts a list:

```yaml
# experiments/configs/m5_factorial_varibad_hypernet_nobonus.yaml
extends:
  - base/base_varibad.yaml
  - base/base_hypernet.yaml
experiment_name: m5_factorial_varibad_hypernet_nobonus
env:
  extends: envs/e_final.yaml
integration: hypernet
exploration_bonus: false
iterations: 100
num_seeds: 8
```

Bases in the list are loaded and merged left-to-right; the experiment file's own fields override all bases. `load_config` handles both the string-form and list-form `extends`.

**Schema validation.**

Every YAML file is loaded into a Python dataclass with type hints (`ExperimentConfig`, `AgentConfig`, `EnvConfig`). If a field is missing or wrong-typed, the script fails at startup with a clear error — not deep into training. This also means autocomplete works when editing the config loader, which catches typos before they become silent bugs.

**What lives in YAML vs what lives in code.**

In YAML: anything a reasonable experiment run might need to vary. Hyperparameters, architecture sizes, network widths, learning rates, env parameters, seed count, iteration count, which ablation axes are on.

In code: things that define a *different* experiment rather than a variant of the same one. Loss function shape, architecture class (MLP vs GRU vs variational encoder), training-loop structure. These go in agent files, not configs.

The rule of thumb: if changing a value should produce a meaningfully different experiment that belongs in a different config file, it's a YAML field. If changing it changes the *science*, it probably needs a new agent file or a code-level decision.

### Version pinning

`pyproject.toml` pins *exact* versions of the core dependencies: `jax`, `jaxlib`, `flax`, `optax`, `chex`, `numpy`. Pins are chosen in M0 based on what works on M4 CPU; any subsequent upgrade is a deliberate decision reflected in a git commit, not a casual `uv sync`. The `uv.lock` file is committed to the repo. A fresh clone + `uv sync` reproduces the exact environment.

The JAX version is particularly important: on M4, the aarch64 CPU wheel has to be chosen, and JAX semantics around `lax.scan` and `vmap` have occasionally shifted between versions. Pin and don't drift.

### Test discipline

Tests exist for load-bearing correctness properties — not full coverage, but the handful of things that, if wrong, silently invalidate research results. Each test file produces a script-output JSON and is run as part of `uv run python -m scripts.make_milestone M0` (and optionally before every long training run).

**Test categories and what each verifies:**

- **Env invariants** (`tests/test_envs.py`): inventory stays within declared bounds across random trajectories; reward is always finite; regime index is always in {0, 1, 2}; done flag set correctly at episode boundary; reset produces the declared initial distribution over states.

- **Belief correctness** (`tests/test_beliefs.py`): output of `beliefs/hmm_posterior.py` matches a brute-force marginalization posterior on short (length ≤ 5) test sequences with known regime ground truth. Posterior always sums to 1 and is non-negative. Sharpens monotonically with consistent evidence.

- **Oracle convergence** (`tests/test_oracles.py`): VI converges (Bellman residual decays below threshold); optimal policy is stable across repeated runs with the same seed; Q-values are finite.

- **Ground-truth leak regression** (`tests/test_leak.py`): for every agent that is *not* supposed to see the regime (everything except Oracle-PPO, Belief-PPO, per-regime PPO), verify that the regime does not appear in the agent's observation input after env.step returns. A simple assertion: construct a rollout, check that the agent's observed tensor has no component that equals the true regime.

- **Run-mode behavior** (`tests/test_run_modes.py`): `--super-fast` on the dummy env completes in <30 seconds; `--fast` in <5 minutes; both produce valid JSON matching the script-output schema.

- **Script output schema** (`tests/test_script_output.py`): every `write_summary` call produces a JSON that validates against the shared schema defined in `utils/script_output.py`.

- **JAX compilation behavior** (`tests/test_jax_perf.py`): runs a short dummy-agent rollout and asserts that the second iteration does not trigger a recompilation (measured via JIT cache hits, or by timing: `per_iter_time` on iteration 2 is within 2× of the median of iterations 2-10). Catches shape-drift bugs before they affect real training runs.

Each test file prints `[test_{name}] OK | tests_run=N | tests_passed=N` on success. CI-style discipline: tests must pass before a milestone is declared passed. They run fast (seconds) and catch real bugs.

### Reproducibility

Each experiment run is fully reproducible from `{config.json, commit hash, seed}`. The commit hash is recorded in `config.json` at run start. If a run produces surprising results, the first question is "what commit was this from?" — which is immediately answerable from the result directory alone.

### Git discipline

- **Main branch always passes M1 at minimum** (ideally whichever milestone is the most recent pass). Never commit to main when the known-good state is broken.
- **Tag each passing milestone.** `git tag m1-passed`, `m2-passed`, etc. Tags are the recovery points — if a later change breaks M2, reverting to `m2-passed` is one command.
- **Branch before risky refactors.** Work on a branch; merge only after the target milestone re-passes. Any change that touches more than one agent or the training loop qualifies as "risky."
- **Commit hash in config.json is the diagnostic tool.** When results change and you don't know why, `git log {old_commit}..{new_commit}` is where the answer is.
- **Never force-push main.** Branches are fine.

**Handling dead ends during env iteration (M2).** Trying parameterizations that fail R1–R4 is expected; the spec's env-design-process is explicitly iterative. Workflow:
- Each env iteration lives on a branch named `env-iter-e{N}`.
- If the env iteration passes R1–R4, merge to main and tag.
- If it fails, commit the attempt + its JSON (the `FAIL.md` documents why it failed), push the branch, then start the next iteration on a new branch. Don't delete failed attempts — they're thesis-appendix material ("we tried X, Y, Z before converging on the final env").
- `envs/e_final.yaml` is a symlink to whichever env config passed. When it changes, every downstream milestone must be re-run.

**Handling milestone re-runs.** Re-running an earlier milestone invalidates all downstream results. Workflow:
- Revert to the earlier milestone's tag (`git checkout m{n}-passed`) or branch from current if the re-run is a forward change.
- Re-run M{n} with `uv run python -m scripts.make_milestone M{n}`.
- If it passes, re-tag (force-move the tag or create `m{n}-passed-v2`). If it fails, document in `FAIL.md`.
- Re-run every downstream milestone. This is expensive, which is why the milestone discipline aims to get each milestone right before moving on — but is unavoidable when fundamental changes are needed.
- Record the re-run in `results/milestones/M{n}/REVISION_LOG.md`: what changed, why, what JSONs were invalidated, what was regenerated.

**Handling experimental dead ends within a milestone.** If you try a parameter combination that doesn't work (e.g., a hypernet hidden-dim that gives flat learning curves), commit the attempt with a descriptive message and keep going. The failed attempt's `stats_*.json` is useful in the thesis appendix as "we tried configuration X and it didn't work for reason Y." Don't silently delete failed attempts; that loses negative-finding information.

### Milestone discipline (recap)

Every milestone produces:
- JSON stats under `results/milestones/M{n}/`
- Figures under `figures/milestones/M{n}/`
- A `PASS.md` or `FAIL.md` note committed after review
- A git tag on pass: `git tag m{n}-passed`

Regenerate everything for a milestone with `uv run python -m scripts.make_milestone M{n}`. A failing milestone blocks downstream work. Milestones are rerun as code evolves; figures always reflect current state.
