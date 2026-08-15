# Plotting conventions

Every figure in this project is written for the thesis, not as a dev-time
scratch plot. These rules apply to every chart under `figures/results/RQ{n}/`
and `figures/appendix/`, and to any new plot you add.

Where a chart goes is not a plotter's decision: register it in
`utils.paths.FIGURE_HOME` and save through `fig_targets(name)`, which returns
its path in the repo tree and in the thesis tree. See `docs/pipeline.md` →
"Figures".

The **canonical implementation** lives in `plotting/style.py` (helpers) and
`plotting/m5r_plots.py` (worked examples). When in doubt, mirror those.

---

## Naming & labels

- **No pipeline jargon** in titles or anywhere visible: drop `M5`, `Step`,
  `RQ1`, `E_final`, phase names, internal sweep IDs, etc. Plain
  professional language only — these will end up in a thesis PDF.
- **Environment naming**: the market-making env is `MarketMakingV1`. Do
  **not** suffix `(E_final)`, do **not** use the legacy name `mm_reduced`,
  do **not** use `MM E_final`. The Avellaneda–Stoikov baseline (M1 only)
  is `Avellaneda–Stoikov baseline`.
- **Method capitalization**: `Concat`, `Hypernetwork`, `Regime-agnostic
  PPO`, `Belief-PPO`, `Oracle-PPO`, `RL²`, `VariBAD`, `PPO floor`,
  `Stacked-obs PPO`. No `concat`, no `hypernet`, no `Hyper-network`.
- **Title states the environment**: every title should make the data
  source unambiguous, e.g. `MarketMakingV1 — reference levels and gap
  decomposition`, `Toy environments — meta-RL methods clear the PPO
  floor`.
- **No rotated y-axis labels.** Don't set `ax.set_ylabel(...)` — the
  90°-rotated text running up the side reads like vertical Compute and
  hurts at-a-glance comprehension. Put the y-axis description as a
  **second line in the title** instead:

  ```python
  ax.set_title(
      "MarketMakingV1 — RL²/VariBAD × Concat/Hypernetwork\n"
      "Final return (mean across n=8 seeds, 200 iter)"
  )
  ```

  Multi-panel figures: use a 2-line `fig.suptitle(...)` and drop
  per-panel ylabels. The x-axis label still uses `ax.set_xlabel(...)`
  (it sits horizontally below the axis, so it does not have the
  rotated-text problem).
- **X-axis labels are capitalized full words**: `Iteration`,
  `Inventory q`, `Timestep within episode`. Not lowercase, not
  abbreviated.

## Legend

Use `LEGEND_OUTSIDE_RIGHT` (or `LEGEND_BELOW`) from `plotting/style.py`:

```python
ax.legend(**LEGEND_OUTSIDE_RIGHT)
fig.subplots_adjust(right=0.78)   # reserve space for the legend
```

Rules:

- **Outside the plot area.** Never let the legend overlap chart data.
- **Opaque white background** with subtle border (`facecolor="white"`,
  `edgecolor="#cccccc"`, `framealpha=1.0`). The chart must remain readable
  through the legend region — no transparency that bleeds.
- **Lists every chart element.** Bars, reference lines, dashed lines,
  chance/random-guess lines, fill bands. Nothing unlabeled. For coloured
  bars use `matplotlib.patches.Patch` proxies.
- Reference-line labels include the numeric value:
  `f"Regime-agnostic PPO floor = {floor:.1f}"`. Use
  `plotting.style.draw_reference_lines(...)` to get this for free.

## Y-axis scaling

- **Bar charts vs references** start at `floor − 5` so bars are
  comparable to the floor reference line.
- **Cap upper bound at `oracle_ceiling + 5`** so the oracle line is
  visible (not pinned to the top edge).
- **Smoke-test rule:** every chart must show *some* visible data even
  when the smoke values fall outside the production y-range. If smoke
  data drops below the floor, extend the bottom:

  ```python
  ymin = min([floor, *all_lower_cis]) − 5
  ```

  Never leave a chart looking empty just because the smoke values
  are tiny.

## Bar value labels

- Every bar carries its mean value as a black numeric label.
- The label must **not** overlap the vertical CI/std cap. Place it
  *inside the bar, just below the lower CI cap*, with a fixed pixel
  offset so the gap holds across data scales:

  ```python
  ax.annotate(f"{mean:.1f}", xy=(x_pos, lo),
              xytext=(0, -3), textcoords="offset points",
              ha="center", va="top", fontsize=9, color="black")
  ```

  Acceptable alternative: above the upper CI cap with ≥6pt offset. Never
  on top of the cap line.

## Compute-budget annotation

Every chart shows the effort behind the stats — italic footer like
*"Compute: 200 iter × 8 seeds × 512 envs × rollout 128"*. Use the helper:

```python
from plotting.style import budget_annotation
budget_annotation(
    fig,
    iterations=int(metrics["iterations"]),
    parallel_envs=int(metrics["parallel_envs"]),
    rollout_length=int(metrics["rollout_length"]),
    num_seeds=int(metrics["num_seeds"]),
    extra="across 12 method×env cells",  # optional context tag
)
```

Pull the values from a real `metrics.json` rather than hard-coding —
this keeps the footer honest when the budget changes (smoke vs full
budget). If the chart is purely analytical (VI / closed form), skip the
footer or pass an `extra=` describing the analytical computation.

## Reference lines

Use `plotting.style.draw_reference_lines(...)`:

```python
draw_reference_lines(ax, floor=136.2, belief=168.5, oracle=180.1)
```

Style: `linestyle="--", linewidth=1.0, alpha=0.6`, colours from the
`COLORS` palette (`ppo`, `belief_ppo`, `oracle_ppo`).

## Layout

- `tight_layout()` first, then `subplots_adjust(right=…, bottom=…)` to
  reserve space for outside legends and the budget footer. Typical
  values: `right=0.65–0.78`, `bottom=0.15–0.30`.
- For multi-panel figures with a shared legend below, push the legend
  far enough below the xlabel: `bbox_to_anchor=(0.5, -0.30)` and
  `subplots_adjust(bottom=0.42)` is a good starting point.
- DPI is set globally by `apply_style()` (savefig dpi 150, tight bbox).
  Always call `apply_style()` at the top of every plot function.

## Quick checklist before committing a figure

- [ ] Title has no `M{n}`, `Step`, `RQ`, or env-version suffix
- [ ] Environment name is `MarketMakingV1` (or `Avellaneda–Stoikov
      baseline` for M1)
- [ ] **No `ax.set_ylabel(...)`** — y-axis description is on a second
      line of the title
- [ ] X-axis label is capitalized full words
- [ ] Legend sits outside the plot area, opaque white, lists every
      element
- [ ] Bars have value labels below the lower CI cap
- [ ] Y-axis honours the floor/oracle convention and is non-empty under
      smoke data
- [ ] Italic compute-budget footer is present and matches reality
- [ ] Reference lines (where applicable) include numeric values in their
      legend labels
- [ ] `apply_style()` was called

## When to update this doc

- A new convention is established (e.g. user feedback on a plot).
- A new shared helper lands in `plotting/style.py`.
- A milestone introduces a chart type that needs its own variant rules.

Keep the doc tight — code-anchored and concrete. Long prose belongs in
research-questions.md, not here.
