# Tinker-Cookbook SFT Logging

Use this file when a research-grade SFT experiment wants to mirror upstream `tinker-cookbook` logging shape without importing that repo into `experiments/`.

Repo-neutral ownership still lives in `scaffolds/README.md` -> `Contract split: who owns what`. This note adds the tinker-cookbook-specific mapping only.

## Design Pillars To Borrow

1. Single log directory for one run
2. Hyperparameter capture once
3. One merged metrics row per step
4. Multiple sinks when needed: JSONL plus optional external loggers
5. Checkpoint registry with machine-readable rows
6. Optional trace / timing sidecars when profiling is the goal
7. Explicit eval timing semantics

## What Upstream Does Well

The supervised stack in `tinker-cookbook` uses:

- one `log_path` directory for machine-readable run outputs
- `log_hparams` for config capture
- `log_metrics(metrics, step)` so train and eval scalars share one step index
- optional console / WandB / Neptune / Trackio sinks
- typed checkpoint records plus `train/checkpoints.jsonl`
- optional `trace_events.jsonl` and `timing_spans.jsonl`
- documented pre-step eval semantics for pipelined supervised training

For this repo's templates, the key resume takeaway is structural:

- same-run resume reads loop state from the matched checkpoint row
- fresh checkpoint loading is a separate weights-only path
- on the current MinT endpoint, same-run resume still uses `create_lora_training_client(...)` followed by `load_state_with_optimizer(...)`
- checkpoint rows may keep one logical `name`, but the runtime save names behind
  `state_path` and `sampler_path` should stay distinct, for example
  `<name>-state` and `<name>-sampler`

## Where Each Concern Lives In This Repo

| Concern | `scaffolds/` | `scaffolds/profiles/sft.md` | Concrete experiment | `$new-experiment-plus` skill |
|---------|--------------|----------------------------|---------------------|------------------------------|
| Baseline eval snapshot + `METRIC` | `scaffolds/README.md` artifact table | — | `write_outputs`, `emit_metric_lines` | scaffold-owned baseline contract |
| SFT JSONL names + truncation policy | `scaffolds/README.md`, `naming.md` | CLI sketch, reset helper note | `reset_supervised_append_streams`, append streams | workflow + self-check |
| Checkpoint registry as restore contract | `scaffolds/README.md`, `naming.md` | checkpoint-row schema, `load_training_state*`, resume semantics | `train/checkpoints.jsonl` rows carry logical `name`, `state_path`, `sampler_path`, and SFT loop state (`step`, next `epoch`, next `batch`) | workflow + self-check |
| Train/eval merge + `test/` prefixes | — | post-step convention | experiment-local merge helper | SFT mapping table in `SKILL.md` |
| Row cadence + RAM discipline | mention in README | cadence flags | `--train-metrics-every`, `--train-print-every` | observability checklist |
| `run.json` provenance | `train.py.tpl` comment | — | experiment `write_outputs` | workflow step on provenance |
| Optional git metadata | — | — | optional experiment helper | this reference |
| AI/script-first run index | promoted only when repo-wide | optional note | experiment-local manifest + stdout marker | workflow step on AI-native index |
| Full config / `code.diff` capture | optional future promotion | — | optional experiment feature | backlog |
| WandB / Neptune sinks | — | — | optional experiment feature | optional sinks guidance |
| Trace / timing files | — | — | only if profiling matters | profiling checklist |

## Already Mirrored In The Repo

| Concern | `experiments/lawbench` | `experiments/fingpt` |
|---------|------------------------|----------------------|
| Streamed `train/metrics.jsonl` with merged `test/eval_*` on eval steps | yes | yes |
| Optional `train/batches.jsonl` (bounded per-step prompt / assistant-text lineage) | yes (first five rows per step) | no (intentional; keeps the SFT log small) |
| `eval/periodic.jsonl` + `train/checkpoints.jsonl` + run-scoped truncation | yes | yes |
| Cadence flags for disk and stdout | `--train-metrics-every` / `--train-print-every` | same |
| `run.json` with `args`, raw `argv`, shell-safe `command` | yes | yes |
| Optional git metadata | yes (unconditional when git is available; no CLI toggle) | no |

Use this table as the "what's already done" reference when upgrading a third
experiment — copy only the rows the new benchmark actually needs.

## Backlog: Promote Only When Shared

Promote these to scaffold-level guidance only when two experiments need the same shape:

- dump full resolved config as `config.json`
- store `code.diff` or a link policy
- promote the same logical-`name` plus split runtime save-name helper only when two experiments share the exact same checkpoint-save shape
- optional WandB / Neptune behind env flags
- optional `trace_events.jsonl` / `timing_spans.jsonl` for latency debugging

## How To Use This File

- When upgrading an experiment with `$new-experiment-plus`, copy only the rows the benchmark actually needs.
- When promoting a pattern to `scaffolds/`, update `scaffolds/README.md`, `naming.md`, `AGENTS.md`, and the relevant skills in the same commit.
