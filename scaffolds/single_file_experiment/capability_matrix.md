# Capability Matrix

How the currently documented bundled experiments map onto the canonical template.

## Capability coverage

| Capability | lawbench | fingpt | chat-dpo | dapo-aime24 |
| --- | --- | --- | --- | --- |
| runtime | mint | mint | mint | tinker |
| `.env` loading | yes | yes | yes | yes |
| tokenizer cache | yes | yes | yes | yes |
| `--dry-run` | yes | yes | yes | yes |
| `--eval-only` | yes | yes | yes | yes |
| eval adapter functions | customized | customized | customized | customized |
| SFT training | yes | yes | no | no |
| DPO training | no | no | yes | no |
| multi-benchmark profile | no | no (uses `--task-type` + CLI data paths) | no | no |
| GRPO training | no | no | no | yes |
| checkpoint / resume | yes | yes | yes | yes |
| `METRIC name=value` output | yes | yes | yes | yes |

## Template alignment status

| Dimension | lawbench | fingpt | chat-dpo | dapo-aime24 |
| --- | --- | --- | --- | --- |
| `run_eval` / eval entrypoint | yes | yes | yes | `evaluate_with_sampler` |
| `run_train` takes client params | yes | yes | yes | yes |
| `compute_eval_metrics` is aggregation | yes | yes | yes | inline in eval |
| `main_async` orchestrates clients | yes | yes | yes | yes |
| eval-only uses sampling client only | yes | yes | yes | yes |
| CLI for eval hyperparams | yes | yes | yes | yes |
| async-first API calls | yes | yes | yes | yes |

## Distillation source

Use these experiments as reference when extending the template:

- `experiments/lawbench` — research-grade SFT after eval, checkpoint + resume, merged periodic eval
- `experiments/fingpt` — `--task-type` driven SFT with the same checkpoint + resume contract as lawbench
- `experiments/chat-dpo` — DPO training on MinT with the same run-dir / checkpoint semantics as the SFT experiments
- `experiments/dapo-aime24` — GRPO, checkpointing, resume

## Convergence target

After aligning experiments to the new template:

- Same core adapter function names where possible; `evaluate_with_sampler` is an accepted variant of `run_eval` in `experiments/dapo-aime24`
- Same `main_async` orchestration pattern (data → client → dispatch)
- Same artifact layout for the baseline set (`run.json`, `console.log`, `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`)
- Benchmark-specific logic stays local (graders, prompts, rewards, metrics)
