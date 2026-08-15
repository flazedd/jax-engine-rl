# The experimental pipeline

One command reproduces everything the thesis reports:

```
uv run python -m scripts.run_matched_programme            # the full programme
uv run python -m scripts.run_matched_programme --dummy    # the same chain on synthetic data, ~3s
uv run python -m scripts.run_matched_programme --dry-run  # the plan and its estimate
uv run python -m scripts.status                           # where a running programme is
```

Read this before touching `scripts/run_matched_programme.py`, adding a stage,
or moving anything under `results/`.

---

## Layout

Directories are named for the part of the thesis they serve. The earlier M0–M7
scheme named them for the milestone that produced them, which is why `M5` sat
beside `M5R` and `milestones/cartpole` beside `M5R/final`: nothing in a name
told you which claim an artifact supported, and seven-week-old results were
adopted as current more than once.

| Path | Holds | Thesis |
|---|---|---|
| `results/foundations/` | environment validation R1–R4, implementation validation | §3.5, §3.8 |
| `results/medium/` | the eight runs on `e_final` | RQ1, RQ2, and RQ3's medium level |
| `results/sweep/` | RSMM difficulty instances | RQ3, §4.1.2 |
| `results/cartpole/` | second domain | RQ3, Appendix B |
| `results/analysis/` | comparison sets, probes, diagnostics, generated tables | Ch. 5 |
| `results/_archive/` | superseded data, kept for provenance comparison | — |
| `figures/results/RQ1\|RQ2\|RQ3/` | charts Chapter 5 includes, under the question they answer | Ch. 5 |
| `figures/appendix/` | charts the appendices include | App. A, B, C |

`medium/` is named for the experiment rather than for one research question
because the same eight runs feed RQ1, RQ2 and RQ3's medium level. Experiments
and research questions are not in one-to-one correspondence, and pretending
otherwise would force arbitrary homes.

Never hardcode these paths. `utils/paths.py` is the single source of truth:
`experiment_dir(name)` routes a run to the right section, `analysis_dir()` and
`foundations_dir()` give the rest. All of them honour the environment overrides
described below.

## Figures

Two destinations exist and only two: `results/<RQ>` for the charts Chapter 5
includes, `appendix/` for everything the appendices carry. The repo's figure
tree and the thesis's figure tree are mirror images, so a chart sits at the same
relative path in both and a thesis include path names the question it answers.

`utils.paths.FIGURE_HOME` is the single definition of that layout, and
`fig_targets(name)` returns both destinations for a chart. A plotter calls it
rather than joining a directory itself:

```python
for out_path in fig_targets("m5r_method_ladder.png"):
    fig.savefig(out_path)
```

An unregistered figure raises rather than landing somewhere plausible. That is
deliberate: the previous flat layout let a chart appear in the thesis directory
without anyone choosing to put it there.

Publication is driven by the `\includegraphics` paths in the document, so a
figure the thesis stopped including stops being copied. The document root is
resolved independently of the figure root, because deriving it from the figure
root made a dummy run look for `.tex` files inside `figures_dummy/`, find none,
and publish nothing at all.

## Phases

The plan runs in dependency order, and the ordering is deliberate: anything
cheap that can invalidate the run comes first.

1. **audit** — config fairness. Verifies inputs, optimiser settings, budget and
   capacity are common across methods. Gates everything.
2. **foundations** — environment validation (R1–R4) and implementation
   validation. R2 and R3 are measured on *trained* agents, so they move when
   the agents change; they are not fixed inputs.
3. **medium** — the eight runs, then the reference-ordering gate, then the
   analyses. The gate requires a positive lower bootstrap bound on each
   adjacent paired reference difference, because the gap-closed fraction
   divides by `oracle − agnostic` and is meaningless if that ordering fails.
4. **sweep**, **cartpole** — RQ3 replication.
5. **figures** — run as soon as their inputs land, not at the end, so the
   thesis fills in while the rest of the programme is still training.

## Provenance

Three mechanisms, each added after the failure it prevents actually happened.

- **Per-seed resume.** Training persists each seed as it finishes. A crash at
  seed 18 of 20 costs one seed, not eighteen. The cache records the budget it
  was produced at and refuses to be reused across a mismatch, because a
  `--super-fast` smoke run otherwise leaves a two-iteration seed behind that a
  full run silently adopts.
- **Provenance stamping.** Training writes `provenance.json` with a fingerprint
  of the resolved agent config. The programme's completeness check requires it:
  budget is not provenance, and a directory trained at the right seed and
  iteration counts under a *superseded architecture* satisfies every numeric
  check.
- **The contract audit.** `scripts.thesis_contract` checks produced artifacts
  against `evaluation/protocol.py`, which holds the constants of Table 3.5 and
  Table D.8. Run it with `--strict` after a programme finishes.

## The dummy layer

`--dummy` writes plausible synthetic results for every stage and renders the
real figures from them, in about three seconds. Its purpose is to prove the
analysis and figure chain works *before* spending a day of training on it; it
has caught several bugs that would otherwise have surfaced only at the end of a
full run.

- Dummy artifacts live beside real ones as `<stem>.dummy.json`, so one listing
  shows what has landed. Figures go to `figures_dummy/`.
- `utils.paths.resolve_data()` is the source switch: a dummy run reads the
  sibling where one exists, a normal run never looks at one.
- Every dummy figure is watermarked and every dummy table carries a banner row.
  Check existence on the *resolved* path, never the raw one — that mistake has
  been made in six modules.
- It does **not** exercise training. `--super-fast` and `--fast` do that.

## Environment overrides

| Variable | Effect |
|---|---|
| `THESIS_RESULTS_ROOT` | where results are read and written |
| `THESIS_PROJECT_FIGS` | the repo's figure tree |
| `THESIS_FIG_ROOT` | the thesis figure tree |
| `THESIS_DUMMY=1` | dummy mode: source switch, watermarks, banners |
| `THESIS_FIG_STAMP=0` | suppress the timestamp on figures, for a final build |

## Adding a stage

Declare it in `build_plan()` with the file it produces, its dependencies and a
rough estimate. The driver skips a stage whose output is current, validates the
artifact the moment it lands, and isolates failures so a broken stage costs
itself rather than the programme.

Three rules learned the hard way:

- **Every figure the thesis includes needs a stage.** Three did not, and a
  clean-room run would have trained for days before failing at the figure.
- **A stage's declared output must be where the script actually writes.** The
  driver treats a mismatch as failure, which is correct and confusing. Declare
  figure outputs with `_repo_fig(name)` so the stage and the plotter resolve the
  same path from `FIGURE_HOME`.
- **Publication runs after the last stage, in both modes.** It used to be
  skipped for dummy runs, so the charts produced by the final stages never
  reached the thesis tree and the dummy chain never exercised the step it
  exists to prove.
