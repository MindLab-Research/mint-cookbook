# Artifact Mapping

Use this file when the user wants a figure from experiment artifacts rather than from a finished result table.

## 1. Pick one source of truth per curve

Do not mix logs with different cadence or semantics into one line unless the relationship is explicit.

Common mapping:

- dense step log: optimization curves
- sparse periodic eval log: benchmark-over-training curves
- checkpoint registry: sparse markers only
- run metadata: display labels, hyperparameters, subtitle text

If one figure is derived from one concrete run directory, prefer writing the exported figure files under that run's local `figures/` directory. Use a shared output directory only when the user explicitly wants a cross-run or cross-project figure bundle.

## 2. Prefer semantic clarity over convenience

If the training log contains merged eval values and a separate periodic eval file also exists, prefer the periodic eval file for the headline benchmark curve unless there is a strong reason not to.

Why:

- one eval event usually equals one row
- the x-axis cadence is easier to interpret
- the chart does not inherit arbitrary train logging cadence

## 3. Typical training-curve mapping

For optimization figures:

- x-axis: step, epoch, token count, wall-clock time, or sample count
- y-axis: one optimization metric per panel
- common metrics: train loss, reward, throughput, gradient norm

For benchmark-over-training figures:

- x-axis: the training progress unit used when eval was triggered
- y-axis: one benchmark metric
- annotate best or final checkpoint only when that is part of the story

## 4. Multi-run comparison rules

When multiple runs are compared:

- keep the same x-axis unit across runs
- normalize labels before plotting
- do not use raw directory names as titles
- move long run identifiers into subtitles or captions

## 5. Confidence bands and repeated runs

Only draw confidence bands when the data actually represents repeated runs or real uncertainty estimates.

Use:

- mean line plus band for repeated seeds
- single line only for a single run

Do not fabricate statistical-looking bands from a single noisy trace.
