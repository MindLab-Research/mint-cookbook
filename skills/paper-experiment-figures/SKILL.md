---
name: paper-experiment-figures
description: Create publication-ready experiment figures for top-tier conference and journal papers. Use when Codex needs to draw or redraw training curves, evaluation curves, benchmark comparisons, ablation bars, task breakdowns, delta plots, or compact multi-panel figures from experiment artifacts, JSONL logs, CSV tables, or paper result tables. Prefer this skill when the figure must balance paper-grade rigor, clean statistical semantics, vector export, and fast visual scanning.
---

# Paper Experiment Figures

Use this skill to turn experiment outputs into figures that read like strong ML systems paper figures rather than dashboard screenshots or research-blog sketches.

Blend two strengths:

- editorial clarity: one clear takeaway per figure, strong hierarchy, high scan speed
- paper rigor: faithful metric semantics, disciplined axes, compact layouts, vector export, no decorative ambiguity

Do not add decorative elements that do not improve comprehension.

Read `references/figure-recipes.md` when choosing a figure family or layout. Read `references/artifact-mapping.md` when the source data comes from experiment logs rather than a clean result table.

## Workflow

1. State the figure question first.
   - Decide what single claim the figure must support.
   - If the figure is trying to prove two different things, split it.

2. Identify the source of truth.
   - Use the artifact or table with the cleanest semantics.
   - Prefer benchmark-only logs for headline eval curves and step logs for optimization curves.
   - Do not mix sparse eval metrics into dense train curves unless the merge semantics are explicit and useful.

3. Choose the simplest figure family that answers the question.
   - training dynamics: line chart
   - benchmark comparison: grouped bars or horizontal ranking bars
   - delta vs baseline: sorted diverging bars
   - many related views: 2x1 or 2x2 multi-panel, not a dashboard wall
   - when optimization and benchmark curves do not need to be read together, split them into separate figures instead of forcing a combined panel

4. Assign semantic roles before colors.
   - main method or main run: one highlight color
   - baselines or context runs: neutral gray or quieter distinction colors
   - reference lines, checkpoints, or confidence bands: secondary styling only
   - do not color by arbitrary order

5. Build the layout around readability.
   - leave real space for title, subtitle, legend, and annotations
   - avoid axis label collisions and clipped tick text
   - keep legends small; prefer direct labels when there are few lines
   - use panel labels only when the figure is truly multi-panel paper content
   - in a single-metric single-panel export, do not repeat the same experiment or metric phrase as both the figure title and an inner panel title
   - shorten, wrap, or move long identifiers rather than shrinking text until crowded

6. Export like a paper figure.
   - export SVG and PNG by default
   - keep SVG as the source-quality artifact for paper assembly
   - verify the PNG at normal viewing size for overlap, crowding, and line visibility
   - when the figure is derived from one specific artifact run, prefer storing the exports under that run's local figure directory unless the user asks for a shared output location

## Figure selection rules

- Use line charts for optimization and evaluation over steps, epochs, tokens, or samples.
- Use grouped bars only when the category count is small and pairwise comparison matters.
- Use horizontal bars for rankings or leaderboard-style snapshots.
- Use sorted delta bars for per-task gains or losses against a baseline.
- Avoid heatmaps when a reader needs to answer a fast comparison question.
- Avoid radar charts unless the user explicitly asks for them.
- Avoid mixing metrics with different scales on one axis unless the transformation is obvious.

## Style system

- Default background: white
- Default text: black
- Axes: dark gray
- Grid: subtle gray, low opacity
- Prefer Inter, then Helvetica, Arial, sans-serif
- Use thin lines, small markers, and generous whitespace
- Prefer shorter titles and move long identifiers into subtitles or captions
- Use one highlight palette and keep supporting series quiet

Good defaults:

- highlight colors: `#FFFD38`, `#29FD2E`, `#2DFFFE`, `#FD28FC`, `#0B24FB`, `#FC0D1B`
- quiet support colors: `#6B7280`, `#9CA3AF`, `#D1D5DB`
- distinction colors for secondary but meaningful context: `#807F17`, `#0F7F12`, `#11807F`, `#7F0F7E`, `#020C7E`, `#7E0308`

Use the stronger palette for the key claim, not for decoration.

## Chart semantics

- Do not invent uncertainty bands when there are not multiple runs or real variance estimates.
- Do not smooth away unstable behavior unless the user explicitly asks for smoothing and the raw line remains inspectable.
- Do not crop the x-axis to hide warmup or collapse.
- If a figure highlights a best checkpoint, mark it clearly and state the criterion.
- If baseline metrics come from a different dataset slice or evaluation protocol, say so in the subtitle or caption.

## Multi-panel rules

- Use a multi-panel layout only when panels share a common reading story.
- Keep shared x-axis or shared legend when possible.
- Keep each panel on one metric family.
- Do not default to a combined training-plus-benchmark panel when the two plots will be read and cited separately.
- For training-curve figures, a strong default is:
  - left panel: optimization curve
  - right panel: benchmark curve
- For comparing two experiments of the same algorithm, a strong default is:
  - one row per experiment
  - one column per metric family

## Validation checklist

- Is the figure answering one clear question?
- Is the chosen chart type the shortest path to that answer?
- Are title, subtitle, legend, and annotations separated cleanly?
- If the title is long, did you move extra detail into the subtitle or caption instead of forcing overlap?
- If a top legend competed with the title block, did you move it inside the axes or to a corner?
- Are the metric semantics faithful to the artifact source?
- Does the exported PNG still read cleanly at normal size?
- Would the figure remain understandable in grayscale or for a colorblind reader?

## Output expectations

- Prefer one strong figure over a dense dashboard.
- Export SVG and PNG when practical.
- Keep code reusable across runs of the same algorithm family.
