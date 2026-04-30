# Single File Experiment Template

This directory contains the one canonical `train.py` scaffold for the repo.
Every new experiment starts from `train.py.tpl` by customizing five task-specific adapter functions, then optionally adding a training extension via profiles.

The goal is not identical experiments. The goal is that every experiment starts from the same small, readable baseline:

- same adapter function signatures
- same `main_async` orchestration pattern
- same dry-run and eval-only semantics, with explicit `--eval-data` in commands and wrappers; standard single-dataset experiments conventionally pass `data/eval/smoke.jsonl` for `--dry-run` validation and `data/eval/full.jsonl` for the frozen benchmark, while multi-dataset experiments may keep a documented dataset-family layout under `data/`
- same generic SFT baseline already wired into `main_async` (`run_train`, `build_supervised_row_datum`, automatic same-run resume helpers, and shared training CLI flags such as `--rank`, `--learning-rate`, `--num-epochs`, `--max-steps`, `--batch-size`, `--lr-schedule`, `--eval-every`, `--save-every`, `--train-metrics-every`, `--train-print-every`, `--load-checkpoint-path`)
- same default top-level README flow modeled on `experiments/fingpt/README.md` (`Quickstart`, `Fast contract tests`, `Live smoke tests`, `Data`, `Current results`, `References`), with `Current results` starting at `Status: \`placeholder\`` until a checked run exists and switching to `Status: \`checked\`` only when the reported run is actually checked; benchmark contract notes, run-mode notes, checkpoint semantics, and artifact pointers usually fold into `Quickstart` or `Data`
- same measured-run reporting rule: keep eval/train config, result metrics, and wall-clock timing together
- same baseline eval snapshot layout (`run.json`, `console.log`, `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`) and `METRIC name=value` output
- same rule that the eval snapshot is the required minimum, while the generic SFT path only emits append-only train/eval streams when the matching cadence flags are enabled; richer benchmark-specific logging still belongs to `$new-experiment-plus`; see `../README.md` artifact contract
- same default dependency chain in `pyproject.toml.tpl`: `mindlab-toolkit` + `tinker==0.15.0` + `transformers`; the dependency pin is still published as `tinker==0.15.0`, but the scaffold code uses `mint` runtime names by default

Companion templates in this directory provide the rest of the experiment shell:

- `README.md.tpl`
- `autoresearch.sh.tpl`
- `autoresearch.md.tpl`
- `pyproject.toml.tpl`
- `env.tpl` (source template for a gitignored local `.env`; the repo-root `.env.example` remains the only checked-in live env template)
- `data/sources.yaml.tpl`

See `naming.md` for naming rules and `capability_matrix.md` for how the current experiments map onto the template.
