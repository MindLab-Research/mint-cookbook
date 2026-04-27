# Figure Recipes

Use this file when choosing a concrete figure pattern after the claim is known.

## 1. Training dynamics

Use for optimization progress over step, epoch, token, or sample count.

- x-axis: training progress unit
- y-axis: one optimization metric per panel
- common metrics: train loss, reward, throughput, gradient norm
- show raw checkpoints or important events with sparse markers only

Good defaults:

- one line per run
- one highlight line, others quiet
- no confidence band unless multiple runs exist
- use per-point markers only when the eval cadence is sparse enough to stay readable; otherwise keep the line clean and reserve markers for selected checkpoints
- optional inset only when the early regime contains the main story
- if optimization and benchmark are discussed separately, prefer separate exported figures rather than a forced 1x2 layout

## 2. Evaluation curve

Use for benchmark metrics measured periodically during training.

- x-axis: training step or token count
- y-axis: benchmark metric
- source: periodic eval log, not the dense train log if a cleaner eval stream exists

Good defaults:

- separate panel from train loss
- show checkpoint markers only if model selection is part of the story
- annotate best and final only when the figure remains uncluttered
- if the benchmark curve is the headline result, export it as its own figure even when a matching train-loss plot also exists

## 3. Grouped comparison bars

Use for a small set of systems across a small set of benchmarks.

- keep category count low
- sort by the main story when possible
- prefer horizontal bars when labels are long

Avoid when the result table is wide or when the user really wants a trend over time.

## 4. Delta bars

Use for per-task improvement or regression against a baseline.

- compute delta first
- sort by magnitude or by task order if domain order matters
- use a diverging zero-centered axis

This is often better than grouped bars for ablations because it shows gain and loss direction immediately.

## 5. Compact paper multi-panel

Use when one figure must show a full story but still fit paper space.

Strong templates:

- 1x2: optimization plus benchmark
- 2x1: overview plus task delta
- 2x2: two experiments by two metric families

Keep legends shared when possible. Do not repeat axis labels that can be shared cleanly.
Only choose these layouts when the joint reading story is stronger than separate single-figure exports.

For a single-panel export, avoid a redundant inner panel title when the figure title already names the experiment and metric.

## 6. Style presets

### Editorial-paper hybrid

Use when the goal is fast visual scanning with paper-safe discipline.

- white background
- black text
- thin lines
- subtle grid
- one bright highlight color
- quiet context lines
- strong contrast and fast scan speed over decorative polish

### Strict paper

Use when the surrounding paper style is conservative.

- white background
- mostly grayscale
- color only for the main method
- minimal annotations

### Appendix analysis

Use for denser supporting figures, not the main headline figure.

- can include more runs
- can include throughput or auxiliary metrics
- keep exact semantics explicit

## 7. Layout safety

Treat text collisions as a correctness bug, not a cosmetic nit.

Rules:

- title, subtitle, legend, annotation boxes, and axis labels must not overlap
- if the title is too long, shorten the display title and move the full identifier into a subtitle or caption
- if a top legend competes with the title block, move the legend inside the axes or into a corner
- keep annotation boxes off major lines when possible, and give them a light background if they must sit over data
- prefer larger margins and fewer words over tighter spacing
- verify the PNG layout, not only the SVG layout
