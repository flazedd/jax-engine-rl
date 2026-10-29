# Frozen-checkpoint return and decision-loss analysis

This analysis does not train or update an agent. It replays the 512 original evaluation episodes
for each of the 80 learned checkpoints and checks every mean return against the published data.
Prepare the saved data and checkpoints using REPRODUCE.md, then run from the repository root:

```sh
python -m scripts.decision_loss_analysis
python -m scripts.summarize_decision_loss --input results/analysis/decision_loss/summary.json --tables /tmp/decision-loss-tables --figures /tmp/decision-loss-figures
```

To regenerate the presentation of the saved analysis without replaying checkpoints:

```sh
python -m scripts.summarize_decision_loss --input reproduction/data/decision_loss_summary.json --tables /tmp/decision-loss-tables --figures /tmp/decision-loss-figures
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

Main explanation charts can be regenerated from the same saved summary:

```sh
python -m scripts.plot_control_explanations --input reproduction/data/decision_loss_summary.json --output /tmp/control-charts
```
