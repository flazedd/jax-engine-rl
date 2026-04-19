# Progress milestones — index

The project progresses through gated milestones M0 through M7. Each milestone produces verifiable JSON artifacts and blocks downstream milestones until passed.

## How to use this folder

- Read `overview.md` once at the start to understand the dependency graph, milestone template, and RQ-to-milestone mapping.
- When actively working on a milestone, read only that milestone's file (e.g., `m2.md`) plus `contingency.md`.
- Don't load all milestones at once — they're 60k+ characters combined. Per-file loading keeps Claude Code's context focused.

## Files

- `overview.md` — dependency graph, RQ/milestone/figure table, total project shape, "when is the thesis done?", milestone template, principles.
- `m0.md` — Infrastructure skeleton.
- `m1.md` — Environment plumbing and AS baseline.
- `m2.md` — Regime-switching env and R1–R4 verification.
- `m3.md` — Reference-level decomposition (answers RQ1).
- `m4.md` — Implementation validation suite.
- `m5.md` — Method ladder on MM (answers RQ2).
- `m6.md` — Difficulty sweep and posterior-performance relationship (answers RQ3).
- `m7.md` — Supplementary ablations (optional).
- `contingency.md` — Failure-mode playbooks, one section per milestone.

## Blocking structure

```
M0 → M1 → M2 → M3 ─┐
              │    ├── answers RQ1
              ▼    │
              M4   │
              │    │
              ▼    │
              M5 ──┴── answers RQ2
              │
              ▼
              M6 ──── answers RQ3
              │
              ▼
              M7 (optional)
```

A failing milestone blocks all downstream work. Re-running an earlier milestone invalidates all downstream results.
