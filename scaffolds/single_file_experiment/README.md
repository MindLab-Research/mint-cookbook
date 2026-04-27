# Single File Experiment Template

This directory contains the one canonical `train.py` scaffold for the repo.
Every new experiment starts from `train.py.tpl` by customizing five task-specific adapter functions, then optionally adding a training extension via profiles.

The goal is not identical experiments. The goal is that every experiment starts from the same small, readable baseline:

- same adapter function signatures
- same `main_async` orchestration pattern
- same dry-run and eval-only semantics
- same generic SFT baseline already wired into `main_async` (`run_train`, `build_supervised_row_datum`, automatic same-run resume helpers, and shared training CLI flags such as `--rank`, `--learning-rate`, `--num-epochs`, `--max-steps`, `--batch-size`, `--lr-schedule`, `--eval-every`, `--save-every`, `--train-metrics-every`, `--train-print-every`, `--load-checkpoint-path`)
- same default top-level README flow modeled on `experiments/fingpt/README.md` (`Quickstart`, `Live smoke tests`, `Data`, `Current results`, `References`), with benchmark contract notes, run-mode notes, checkpoint semantics, and artifact pointers usually folded into `Quickstart` or `Data`
- same measured-run reporting rule: keep eval/train config, result metrics, and wall-clock timing together
- same baseline eval snapshot layout (`run.json`, `console.log`, `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`) and `METRIC name=value` output
- same rule that the eval snapshot is the required minimum, while the generic SFT path only emits append-only train/eval streams when the matching cadence flags are enabled; richer benchmark-specific logging still belongs to `$new-experiment-plus`; see `../README.md` artifact contract
- same default dependency chain in `pyproject.toml.tpl`: `mindlab-toolkit` + `tinker==0.15.0` + `transformers`, while the scaffold code still uses `tinker` runtime names until the repo promotes a different default

Companion templates in this directory provide the rest of the experiment shell:

- `README.md.tpl`
- `autoresearch.sh.tpl`
- `autoresearch.md.tpl`
- `pyproject.toml.tpl`
- `env.tpl`
- `data/sources.yaml.tpl`

See `naming.md` for naming rules and `capability_matrix.md` for how the current experiments map onto the template.
