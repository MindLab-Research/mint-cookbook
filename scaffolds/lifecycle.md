# Lifecycle-First Architecture

## Core Idea

A new experiment begins by defining the benchmark contract, not by choosing a training algorithm.

The benchmark contract includes:

- eval dataset and split
- prompt shape
- parser and grader behavior
- primary metric
- output contract: baseline eval artifacts first, research-grade logs only when the experiment explicitly upgrades

Only after these are stable should the experiment add training.

## Canonical Sequence

1. Identify the target method. If starting from only a benchmark, default to the latest benchmark-tested method.
2. Copy the template from `scaffolds/single_file_experiment/`.
3. Read `scaffolds/profiles/eval.md`.
4. Customize the five task adapter functions.
5. Run `uv run train.py --dry-run --eval-data <smoke_eval_path>` to validate data and prompt formatting without credentials.
6. Run `uv run train.py --eval-only --eval-data <full_eval_path>` to establish the frozen benchmark baseline.
7. Decide whether the experiment stays eval-only or adds training.
8. If training is needed, choose exactly one extension: `scaffolds/profiles/sft.md`, `scaffolds/profiles/dpo.md`, or `scaffolds/profiles/grpo.md`.

## Why This Lifecycle

If experiments start by choosing SFT vs DPO vs GRPO, the repo drifts into incompatible shapes.
If they all start eval-first and share the same adapter signatures, the repo can keep one template family, one naming scheme, and one artifact contract.

## Taxonomy

- `eval` - default startup profile for every experiment
- `sft` - training extension added after eval is stable
- `dpo` - pairwise preference training extension added after eval is stable
- `grpo` - training extension added after eval is stable

This is a lifecycle, not four unrelated experiment families.
