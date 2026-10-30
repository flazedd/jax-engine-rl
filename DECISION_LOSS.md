# Frozen-checkpoint return and decision-loss analysis

This analysis does not train or update an agent. It replays the 512 original evaluation episodes
for each of the 80 learned checkpoints and checks every mean return against the published data.
Start with the README installation (`uv sync --locked`) and prepare the saved data and
checkpoints using REPRODUCE.md. Run these commands from the repository root:

```sh
uv run python -m scripts.decision_loss_analysis
uv run python -m scripts.summarize_decision_loss --input results/analysis/decision_loss/summary.json --tables /tmp/decision-loss-tables --figures /tmp/decision-loss-figures
```

To regenerate the presentation of the saved analysis without replaying checkpoints:

```sh
uv run python -m scripts.summarize_decision_loss --input reproduction/data/decision_loss_summary.json --tables /tmp/decision-loss-tables --figures /tmp/decision-loss-figures
```

Verify the saved summary and analysis sources against reproduction/data/decision_loss_manifest.json.
Replay caches in results/analysis/decision_loss record checkpoint, config, collector and script hashes.
A cache with changed inputs is rejected; use a fresh output directory after source changes.

Return accounting uses reward plus the post-fill inventory penalty to recover trading gains.
The finite-horizon dynamic programme uses the undiscounted 128-decision objective and the same
fill/boundary rules. Decision loss is optimal full-information value minus the value of the
sampled action followed by optimal continuation. It is a scoring reference, not information
supplied to the agent.

Own-trajectory loss includes state-visitation differences. Matched-state comparisons condition
on exact decision number, inventory and regime, weighting shared cells by the smaller visit
count within each trained seed pair. Report the overlap: unsupported cells are excluded.
Different histories within a matched cell remain possible. Switch windows end at the next
switch or episode end; initial regime draws are not switches. The first decision under a new
regime precedes feedback from that regime.

Differences use 10,000 paired bootstrap resamples with RNG seed 0. Intervals are exploratory,
unadjusted 95% intervals; no causal architecture or detection-speed claim follows from them.
Expected cumulative decision loss equals the full-information expected return gap, but finite
sample realised returns differ through initial draws and transition/reward sampling noise.
The tests verify the Bellman values against the existing oracle and this telescoping identity.

The complete saved-result workflow now includes these analyses. To regenerate only the main
explanation charts and the current appendix comparison table:

```sh
uv run python -m scripts.summarize_quote_quality --input reproduction/data/decision_quality_comparisons.json --table /tmp/decision_quality_comparisons.tex
uv run python -m scripts.plot_control_explanations --input reproduction/data/decision_loss_summary.json --output /tmp/control-charts
```

The joint switch overview reads the original saved-results archive and verifies all checkpoint hashes and returns against the decision-loss summary:

```sh
uv run python -m scripts.plot_switch_overview --archive reproduction/data/saved-results.tar.gz --decision-loss reproduction/data/decision_loss_summary.json --output /tmp/switch_overview.pdf
```

Four decision-quality comparisons use the cached state-action counts (no checkpoint replay):

```sh
uv run python -m scripts.compare_decision_quality --cache results/analysis/decision_loss --output results/analysis/decision_loss/decision_quality_comparisons.json --table /tmp/decision_quality_comparisons.tex
```

The main quote-quality chart reads the `common_states` configuration levels in this JSON.
For each seed, all four configurations use identical states and the minimum visit count
across all four as weights. The JSON records per-seed levels, coverage, paired reductions,
and 10,000 seed-bootstrap intervals. Pairwise comparisons remain supporting checks.
The return-component chart shows configuration levels for trading gains and penalties.
Positive reductions in the supporting tables favour the first named agent.

The saved quote-quality JSON includes 20 common-sample records, one per training seed, with a
loss and coverage for every configuration. The summary command verifies the saved configuration
means, paired differences, coverage and bootstrap intervals from these records. It does not
reconstruct raw visits. The cache command above rematches raw counts and records their hashes.

The reproduction report checks the analysis manifests and records the source files actually
used, including local changes. Historical commits identify earlier releases rather than the
current chart presentation.
