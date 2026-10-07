# Published experiment data

These files make the repository sufficient to check the reported results and replay the saved
agents. They contain synthetic experiment data; no market-data subscription is needed.

| File | Contents |
| --- | --- |
| `saved-results.tar.gz` | Evaluation returns, learning curves, resolved settings, probe outputs, analytical reference curves, diagnostics, statistical comparisons, and validation results |
| `checkpoints.tar.gz` | All 160 main checkpoints plus saved checkpoints for supporting validation experiments |
| `manifest.json` | SHA-256 hashes of both archives and every file inside, plus the original training commit |
| `m5r_recovery_audit.json` | Verified replay of all 80 main runs: extended recovery bins, direct posterior predictions, and fitted-reference comparisons |
| `m5r_full_belief_mlp.json` | Exploratory MLP probes of VariBAD's means and full policy input, using 500 episodes for each of 40 saved runs |

Files extract relative to a `results/` directory. The quick-start command handles extraction
into an isolated output directory. For direct checkpoint work, run from the repository root:

```bash
uv run python -m scripts.prepare_results --checkpoints
```

Without `--checkpoints`, only the saved measurements are unpacked. The command checks hashes
before writing and refuses to replace differing local files. The compressed archives remain
unchanged when analyses are rerun.

The recovery plot uses the separate `m5r_recovery_audit.json` file to extend the original
`20+` bin through step 126. To regenerate it after extracting checkpoints, run
`uv run python -m scripts.m5r_recovery_audit`. The replay checks both probe types against
the original overall accuracies and all seven original recovery bins before saving results.

The full-belief table combines the original linear results with `m5r_full_belief_mlp.json`.
`scripts.summarize_m5r_full_belief` verifies both classifiers' mean-only scores against the main
probe results and checks matching direct-posterior curves across the two replays. It reports
paired bootstrap intervals as exploratory comparisons, without multiplicity correction.
To rerun the MLP fits after extracting checkpoints:

```bash
uv run python -m scripts.m5r_supplemental_belief_checks --classifier mlp \
  --methods varibad_concat varibad_hypernet --n-rollouts 500 \
  --output results/analysis/m5r_full_belief_mlp.json
```

The command saves each completed run and resumes compatible existing output.

The original implementation baseline uses three seeds; the separate factorial implementation
checks use eight. Main comparisons use 20 seeds per condition. These are separate experiments,
not additional seeds for the main comparison.

The original training commit identifies the training implementation. The Git commit containing
these files identifies the reproduction code; the manifest identifies the exact published data.
The `source_commit_at_packaging` field records the base commit at packaging time, not a claim
that all reproduction changes were already part of that commit.

Maintainers can rebuild these archives from verified local results with
`uv run python -m scripts.build_repository_data`. Review changes to the manifest and run the
saved-result reproduction before publishing updated data.
