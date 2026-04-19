# M0 — PASS

**Date:** 2026-04-19
**Commit:** tagged `m0-passed`

## Pass criteria (from `docs/milestones/m0.md`)

Verified via `results/milestones/M0/stats_M0_pipeline.json`:

| Criterion | Value |
|---|---|
| `all_scripts_exit_zero` | true |
| `all_scripts_wrote_expected_json` | true |
| `super_fast_duration_seconds` | 0.218 (< 30) |
| `fast_duration_seconds` | 0.153 (< 300) |
| `schema_validates` | true |
| `make_milestone_script_succeeded` | true |

Figure `figures/milestones/M0/fig_M0_dummy_learning_curve.png` also rendered (random-policy return is ~flat per iteration, as expected).

## Reproduction

```
rm -rf results figures
uv run python -m scripts.make_milestone M0
```

Exit 0 and the final stdout line starting `[make_milestone] OK | pass=True` are equivalent to a pass.

## What M0 established

- `uv sync` produces a working environment pinned in `pyproject.toml` + `uv.lock` (jax 0.4.38, flax 0.10.2, optax 0.2.4, chex 0.1.88, Python 3.12).
- Config pipeline: YAML with `extends:` composition → `ExperimentConfig` dataclass → `apply_run_mode()` overrides → JSON snapshot in `results/{experiment}/config.json` with commit hash.
- Shared-schema script output: every script calls `write_summary()` and prints one final `[script] OK | ... | output=...` line.
- `lax.scan` + `vmap` rollout is shape-stable and JIT-compiled; `DummyAgent` + `DummyEnv` exercise the full pipeline without research logic.
- Plotting stack loads JSON (never in-memory state) and produces PNGs under `figures/milestones/M{n}/`.
- Orchestrator `scripts/make_milestone.py M0` runs super-fast → fast → full, regenerates figures, validates every summary JSON against the shared schema, writes `stats_M0_pipeline.json`.

## Known non-issues

- `super_fast_duration_seconds` (~0.22s) is slightly larger than `fast_duration_seconds` (~0.15s) because super-fast triggers the first JIT compilation; subsequent shapes reuse warm cache entries. Both are well under the spec thresholds.
