# Frozen-checkpoint inventory exposure and accepted fills

No training or policy updates. First install the locked dependencies with `uv sync --locked`
and prepare the saved data and checkpoints using REPRODUCE.md. Run these commands from the
repository root:

```sh
uv run python -m scripts.inventory_activity_analysis
```

This replays the same 512 evaluation episodes for each of the 80 learned checkpoints.
Each mean return and inventory penalty must match the saved decision-loss analysis
(atol 1e-10). Post-fill inventory is recorded before terminal reset, including the
last decision. Inventory distributions use absolute inventory 0 through 5.

Accepted bid and ask fills are recovered separately from inventory change and
spread earnings: dq=bid-ask, gain=bid_distance*bid+ask_distance*ask. These two equations
uniquely determine both flags, including two accepted fills with no inventory change.
Rejected attempts at the inventory bounds are excluded. A test checks actual terminal
simulator transitions with simultaneous raw fills and inventory bounds. Each replay
also checks both flags are binary, blocked fills are zero, and reward is reconstructed.

Each run contributes equally. Confidence intervals use 10,000 bootstrap samples of
trained runs with RNG seed 0. Differences preserve training seed pairs; individual
evaluation episodes differ between configurations. The complete per-run inventory
histograms and fill counts are saved, with checkpoint/config/collector/script hashes.

To regenerate the chart and table from packaged data, without checkpoints:

```sh
uv run python -m scripts.plot_inventory_activity --input reproduction/data/inventory_activity_summary.json --figure /tmp/inventory_activity.pdf --table /tmp/inventory_activity.tex
```

Verify source and summary hashes in reproduction/data/inventory_activity_manifest.json.
