# Published experiment data

These files make the repository sufficient to check the reported results and replay the saved
agents. They contain synthetic experiment data; no market-data subscription is needed.

| File | Contents |
| --- | --- |
| `saved-results.tar.gz` | Evaluation returns, learning curves, resolved settings, probe outputs, analytical reference curves, diagnostics, statistical comparisons, and validation results |
| `checkpoints.tar.gz` | All 160 main checkpoints plus saved checkpoints for supporting validation experiments |
| `manifest.json` | SHA-256 hashes of both archives and every file inside, plus the original training commit |

Files extract relative to a `results/` directory. The quick-start command handles extraction
into an isolated output directory. For direct checkpoint work, run from the repository root:

```bash
uv run python -m scripts.prepare_results --checkpoints
```

Without `--checkpoints`, only the saved measurements are unpacked. The command checks hashes
before writing and refuses to replace differing local files. The compressed archives remain
unchanged when analyses are rerun.

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
