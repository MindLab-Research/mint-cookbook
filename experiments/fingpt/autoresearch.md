# Autoresearch: fingpt

`autoresearch.sh` is the canonical automation wrapper for this experiment's current runnable line.
It is not automatically the final report or confirmation command.
This file is the wrapper protocol for search work. Keep environment setup, data preparation, and full reproduction commands in `README.md`.

## Objective

Improve the current held-out sentiment line, not redefine the benchmark contract and not silently switch the wrapper to the Fineval slice.

## Current wrapper

- task type: `sentiment`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- train data: `data/fingpt-sentiment-train/train.jsonl`
- final wrapper eval data: four held-out sentiment datasets (`fpb`, `fiqa-sa`, `tfns`, `nwgi`)
- periodic train eval data: `train-eval-160:data/benchmarks/sentiment/train-eval-160/all/test.jsonl`
- train config: `rank=16`, `num_epochs=1`, `batch_size=32`, `learning_rate=1e-4`, `lr_schedule=linear`
- eval config inside the wrapper: `eval_limit=100`, `eval_max_tokens=256`, `max_concurrent_requests=128`
- cadence: `eval_every=10`, `save_every=10`, `train_metrics_every=1`, `train_print_every=1`
- log path: `artifacts/runs/sft-1epoch-sentiment-qwen3-4b-<timestamp>/`

Run it with:

```bash
bash autoresearch.sh
```

Intentional variants should still go through CLI overrides to this wrapper, for example:

```bash
bash autoresearch.sh --learning-rate 5e-5
bash autoresearch.sh --batch-size 16 --num-epochs 1
bash autoresearch.sh --eval-limit 0
```

## Search signals

- Primary search signal: aggregate `eval_accuracy` from the wrapper run
- Periodic diagnostic signal: aggregate `eval_accuracy` on the `train-eval-160` slice
- Secondary diagnostics: aggregate `eval_weighted_f1`, `eval_macro_f1`, plus per-dataset metrics such as `fpb_accuracy`, `fiqa-sa_accuracy`, `tfns_accuracy`, and `nwgi_accuracy`
- Train-side diagnostics: `train_mean_nll` and step timing trends

## Mutable surface

- sentiment SFT recipe knobs such as `--learning-rate`, `--batch-size`, `--num-epochs`, `--rank`, `--lr-schedule`, `--eval-every`, and `--save-every`
- bounded implementation details in `train.py` that preserve the sentiment eval contract
- prompt formatting, label normalization, and sentiment-data preparation logic when the held-out benchmark meaning stays fixed
- the concrete files that normally move together are `train.py`, the sentiment manifests under `data/`, `data/sources.yaml`, `README.md`, and `autoresearch.sh`

## Frozen contract

- the meaning of aggregate `eval_accuracy`
- the held-out sentiment dataset set used by the wrapper's final post-train eval without an explicit contract change
- the distinction between the wrapper's bounded search eval (`--eval-limit 100`) and a clean full held-out confirmation rerun (`--eval-limit 0`)
- the fact that Fineval remains a separate experiment line inside this directory rather than the current wrapper target

## Recovery and confirmation

- Use the same `--log-path` again when a training candidate is interrupted; `train.py` auto-resumes from the latest recorded `state_path` in that run directory
- Use recorded `sampler_path` entries for later `--eval-only --base-model ...` confirmation reruns
- The wrapper's own final eval is a bounded search signal because it keeps `--eval-limit 100`; for reportable held-out confirmation, rerun promising checkpoints with `--eval-limit 0`
- Clean confirmation can be done per dataset or across all four held-out datasets, for example:

```bash
uv run train.py --eval-only \
  --task-type sentiment \
  --base-model '<sampler_path>' \
  --eval-limit 0 \
  --eval-data "fpb:data/benchmarks/sentiment/fpb/test.jsonl,fiqa-sa:data/benchmarks/sentiment/fiqa-sa/test.jsonl,tfns:data/benchmarks/sentiment/tfns/test.jsonl,nwgi:data/benchmarks/sentiment/nwgi/test.jsonl"
```

## Run budget and stopping

- Do not stop a candidate only because the `train-eval-160` slice is noisy; it is a cheap periodic signal, not the full held-out benchmark
- Use falling `train_mean_nll` plus improving wrapper-side held-out metrics as the main continuation evidence during training
- Stop when the bounded wrapper eval is repeatedly flat or worse, train loss has clearly flattened, or the current wall-clock or step budget is exhausted
- Promote promising checkpoints to full `--eval-limit 0` reruns instead of extending one candidate indefinitely

Keep `autoresearch.sh`, `autoresearch.md`, and `README.md` aligned whenever the wrapper recipe changes.
