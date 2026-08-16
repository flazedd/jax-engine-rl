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
- **No baked-in figure title.** The thesis caption names the figure and
  the environment, so a rendered title only repeats it in a second
  typeface. Do not call `fig.suptitle(...)`, and do not put the
  environment name, the seed count or the metric description into the
  image. Panel titles are the exception: in a multi-panel figure a short
  panel identifier such as `Linear probe` is what tells the two panels
  apart, so keep those, `loc="left"`.
- **Y-axis description goes in `ax.set_ylabel(...)`**, short and
  horizontal-reading, e.g. `Return at the end of training`,
  `Difference in probe test accuracy`. Multi-panel figures with a shared
  y-axis label it on the leftmost panel only.
- **X-axis labels are capitalized full words**: `Iteration`,
  `Inventory q`, `Timestep within episode`. Not lowercase, not
  abbreviated.

## Colour

The figures share one palette. Never fall back to the matplotlib default
cycle, and never introduce a hue that is not on this list.

- **Hue encodes the conditioning architecture**: teal for the
  hypernetwork, slate for concatenation.
- **Shade encodes the method within a hue**: the darker tone is RL², the
  lighter is VariBAD. `#1d7870` / `#7ec8bd` for the teals, `#6b757d` /
  `#c3cad0` for the slates.
- **Amber `#e09f3e`** is the non-variant curve: the analytical posterior,
  or Stacked-obs PPO.
- **Deep navy `#264653`** is reserved for Oracle-PPO.
- When architecture is the *plotted quantity* rather than a channel, as
  in a hypernetwork-minus-concatenation difference, hue is free: stay in
  the teal family and let shade carry the method, so the figure still
  reads as part of the set.
- Confidence bands take their curve's colour at `alpha≈0.13` with a thin
  edge in the same colour. Where more than about four bands would
  overlap, drop them and report seed uncertainty in the table instead.

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

The compute budget is reported in the thesis caption, through the
`\compute{seeds}{iters}{envs}{rollout}` macro, not rendered onto the
figure. `budget_annotation` is kept as a no-op stub so existing call
sites still run; do not add new ones. The historical form was:

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

- [ ] **No `fig.suptitle(...)`**; panel identifiers only, and no
      environment name, seed count or metric text baked into the image
- [ ] Colours come from the palette above, hue for architecture and
      shade for method
- [ ] Y-axis description is a short `ax.set_ylabel(...)`
- [ ] X-axis label is capitalized full words
- [ ] Legend sits outside the plot area, opaque white, lists every
      element
- [ ] Bars have value labels below the lower CI cap
- [ ] Y-axis honours the floor/oracle convention and is non-empty under
      smoke data
- [ ] Compute budget is in the thesis caption via `\compute{...}`, not on
      the figure
- [ ] Reference lines (where applicable) include numeric values in their
      legend labels
- [ ] `apply_style()` was called

## When to update this doc

- A new convention is established (e.g. user feedback on a plot).
- A new shared helper lands in `plotting/style.py`.
- A milestone introduces a chart type that needs its own variant rules.

Keep the doc tight — code-anchored and concrete. Long prose belongs in
research-questions.md, not here.
