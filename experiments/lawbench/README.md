# lawbench

This directory is a self-contained MinT experiment for LawBench.
It evaluates the official LawBench benchmark under one fixed local benchmark contract and keeps a maintained local execution baseline built around `Qwen/Qwen3-4B-Instruct-2507` plus LoRA SFT.

Current runnable scope:

- benchmark: the full 20-task LawBench benchmark, materialized locally as `data/eval/full.jsonl`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- training route: LoRA SFT on the public `DISC-Law-SFT` train artifact, then final official LawBench eval
- primary metric: `METRIC eval_lawbench_avg=...`
- reportable data: `data/eval/full.jsonl`

This experiment does not claim full paper-faithful reproduction of Qzhou-Law or DISC-LawLLM. The official scorer and benchmark contract stay fixed, but the maintained runnable line is a smaller local execution baseline.

## Quickstart

The fastest way to reproduce the current repo-local LawBench workflow is:

1. sync the environment
2. build or download the local eval and train artifacts
3. run the dry-run path with `--eval-data data/eval/smoke.jsonl`
4. run a bounded live eval-only smoke check before the full 10,000-row benchmark rerun
5. run SFT plus final eval, or the canonical wrapper
6. pick a saved `sampler_path` from `train/checkpoints.jsonl` and rerun `--eval-only --eval-data data/eval/full.jsonl --base-model <sampler_path>`

Set up the environment and local credentials:

```bash
cd experiments/lawbench
uv sync
cp ../../.env.example .env  # if needed
# fill in MINT_API_KEY and MINT_BASE_URL in .env, or export them in the shell
```

Prepare local data:

```bash
uv run --no-sync data/eval/download_eval_raw.py \
  --output-dir data/eval/raw/lawbench-official

uv run --no-sync data/eval/build_eval_manifest.py \
  --official-data-dir data/eval/raw/lawbench-official

uv run --no-sync data/eval/build_eval_manifest.py \
  --official-data-dir data/eval/raw/lawbench-official \
  --smoke \
  --output-eval data/eval/smoke.jsonl \
  --output-meta data/eval/smoke.meta.json

uv run --no-sync data/eval/build_train_eval_manifest.py \
  --input-eval data/eval/full.jsonl \
  --output-eval data/eval/train_eval_200.jsonl \
  --output-meta data/eval/train_eval_200.meta.json

uv run --no-sync data/train/download_train_raw.py \
  --output-dir data/train/raw/disc-law-sft

uv run --no-sync data/train/build_train_manifest.py \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Pair.jsonl \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Pair-QA-released.jsonl \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Triplet-released.jsonl \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Triplet-QA-released.jsonl \
  --output-train data/train/full.jsonl \
  --output-meta data/train/full.meta.json
```

Validate local data, prompt rendering, scorer wiring, and train/eval overlap checks without remote credentials:

```bash
uv run train.py --dry-run \
  --eval-data data/eval/smoke.jsonl
```

For the cheapest real live eval-only confirmation, run the dedicated live smoke test first:

```bash
uv run python -m unittest tests.test_train.LiveLawBenchFlowTest.test_eval_only_live_smoke
```

Run the frozen eval-only benchmark:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/full.jsonl \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --log-path artifacts/runs/eval-qwen3-4b-$(date +%Y%m%d-%H%M%S)
```

Run training plus final eval:

```bash
uv run train.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --train-data data/train/full.jsonl \
  --eval-data data/eval/full.jsonl \
  --train-eval-data data/eval/train_eval_200.jsonl \
  --log-path artifacts/runs/sft-1epoch-qwen3-4b-$(date +%Y%m%d-%H%M%S) \
  --rank 16 \
  --num-epochs 1 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --lr-schedule cosine \
  --eval-max-tokens 1024 \
  --max-concurrent-requests 128 \
  --eval-every 10 \
  --save-every 10 \
  --train-metrics-every 1 \
  --train-print-every 1 \
  --mint-timeout 600 \
  --seed 42
```

Run the canonical wrapper:

```bash
bash autoresearch.sh
```

After a training run, inspect `train/checkpoints.jsonl` to find the latest saved `sampler_path`.

Evaluate a saved sampler checkpoint recorded in `train/checkpoints.jsonl`:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/full.jsonl \
  --base-model '<sampler_path>' \
  --log-path artifacts/runs/eval-checkpoint-$(date +%Y%m%d-%H%M%S)
```

For each reportable run, keep the evidence bundle together: `run.json`, `console.log`, `eval/metrics.json`, `eval/predictions.jsonl`, and `train/checkpoints.jsonl` when checkpoints are produced.

### Run modes and restore

Important knobs for interpreting the current practical line:

- `--train-data` defaults to `data/train/full.jsonl`; override it only when you intentionally want a different local train manifest.
- `--train-eval-data data/eval/train_eval_200.jsonl` keeps a periodic train-time proxy slice that does not replace the final full benchmark eval.
- `--max-concurrent-requests 128` is the per-task eval in-flight cap in the canonical wrapper.
- `--eval-max-tokens 1024`, `--num-epochs 1`, `--batch-size 256`, `--learning-rate 1e-4`, and `--rank 16` define the maintained SFT line.
- same-run resume is directory-driven: rerun the same training command with the same `--log-path`, and `train.py` restores the latest resumable `state_path` from `train/checkpoints.jsonl`, rebuilds the deterministic `seed:epoch` shuffle, and continues from the next batch offset instead of replaying the last completed batch.
- `--load-checkpoint-path` is the fresh weight-only start path; it does not reuse optimizer state or the previous run's append-only logs.

## Fast contract tests

This experiment no longer keeps a separate credential-free contract unittest tier. Use the live smoke suite below as the maintained validation path.

## Live smoke tests

Default train-flow validation now uses the real MinT backend instead of mocked helpers:

```bash
uv run python -m unittest tests.test_train
```

This suite exercises the actual `uv run train.py` entrypoint families with LawBench smoke data and a live `MINT_*` connection.
The live smoke path uses the fixed 5-task eval smoke slice and `--max-concurrent-requests 8`.
Use this smoke path for routine live validation; reserve `data/eval/full.jsonl` for actual benchmark confirmation.

Current live coverage:

- `--eval-only` on the smoke eval split
- smoke SFT train on `data/train/smoke.jsonl` plus `data/eval/smoke.jsonl`
- interrupted same-run resume by rerunning the same training command in the same `--log-path`

If you only need the bounded remote eval-only check, run the single test method instead of the whole suite:

```bash
uv run python -m unittest tests.test_train.LiveLawBenchFlowTest.test_eval_only_live_smoke
```

## Data

| Split | Path | Rows | Source | Build command | Reportable use |
| --- | --- | ---: | --- | --- | --- |
| eval smoke | `data/eval/smoke.jsonl` | 5 | fixed 5-task scorer-compatible smoke slice | `uv run --no-sync data/eval/build_eval_manifest.py --official-data-dir data/eval/raw/lawbench-official --smoke --output-eval data/eval/smoke.jsonl --output-meta data/eval/smoke.meta.json` | validation only |
| eval full | `data/eval/full.jsonl` | 10000 | official LawBench raw `<task_id>.json` files | `uv run --no-sync data/eval/build_eval_manifest.py --official-data-dir data/eval/raw/lawbench-official` | final benchmark |
| train smoke | `data/train/smoke.jsonl` | 12 | checked-in smoke SFT slice | checked in | validation only |
| train full | `data/train/full.jsonl` | 285781 | public `DISC-Law-SFT` release | `uv run --no-sync data/train/build_train_manifest.py ... --output-train data/train/full.jsonl` | training |
| train-time eval | `data/eval/train_eval_200.jsonl` | 200 | deterministic balanced slice from `data/eval/full.jsonl` | `uv run --no-sync data/eval/build_train_eval_manifest.py --input-eval data/eval/full.jsonl --output-eval data/eval/train_eval_200.jsonl --output-meta data/eval/train_eval_200.meta.json` | periodic eval only |

Eval row contract:

```json
{"example_id": "...", "task_id": "1-1", "task_name": "...", "cognitive_level": "memory|understanding|application", "prompt": "...", "expected": "..."}
```

Train row contract:

```json
{"example_id": "...", "task_name": "...", "cognitive_level": "memory|understanding|application", "prompt": "...", "assistant_text": "..."}
```

Provenance and split rules:

- eval and train artifacts are separate local workflows with separate raw snapshot directories, download scripts, and build scripts
- `data/sources.yaml` records upstream sources, split rules, raw snapshot paths, and local materialization choices
- train/eval overlap policy: eval `example_id` values must stay disjoint from train `example_id` values, and `--dry-run` plus training startup audit that overlap explicitly
- `data/eval/smoke.jsonl` is the repo-standard 5-row smoke eval slice: one example each from task ids `1-2`, `2-5`, `2-7`, `3-3`, and `3-4`, chosen to cover memory/understanding/application plus single-choice, extraction, generation, multi-choice, and regression scorer paths
- smoke artifacts and `data/eval/train_eval_200.jsonl` are not reportable final benchmark results
- benchmark-native layout exception: the upstream LawBench release is one raw `<task_id>.json` file per task, but this experiment freezes a local JSONL eval manifest before profiling

### Resume versus source duplication

- When you investigate repeated examples after an interrupted run, separate resume bookkeeping from train-artifact contents.
- The same-run resume path uses the latest `state_path` plus completed-step count to recover the next epoch and batch; the smoke resume test covers this path, and the intended behavior is to continue from the next batch offset.
- `data/train/smoke.jsonl` has 12 rows with unique `example_id` values and no exact duplicate `prompt` + `assistant_text` pairs, so it is the preferred split for fast resume debugging.
- `data/train/full.jsonl` is intentionally not deduplicated. In the current checked-in artifact it contains 285781 unique `example_id` values but 5982 exact duplicate `prompt` + `assistant_text` groups; the largest exact-duplicate group repeats 55 times.
- Those repeats come from preserving public `DISC-Law-SFT` source rows during materialization, not from same-run resume. If you need a deduplicated train artifact, create and document a separate build target instead of silently mutating `data/train/full.jsonl`.

### Benchmark contract

Primary metric:

```text
METRIC eval_lawbench_avg=...
```

Companion metrics:

- `METRIC eval_mem_avg=...`
- `METRIC eval_understanding_avg=...`
- `METRIC eval_application_avg=...`
- `METRIC eval_abstention_rate=...`

Prompt, parser, grader, and scorer contract:

- prompt shape: single-turn chat messages built from each eval row's legal prompt text
- parser: eval artifacts keep a human-readable `prediction` with chat-template special tokens stripped, while official scoring still uses the raw sampled assistant text
- scorer: vendored official LawBench scorer under `third_party/lawbench_official/evaluation/`, selected by `task_id`; there is no local per-example fallback scorer
- aggregation: `eval_lawbench_avg` is the macro-average over the 20 official task scores; the cognitive-level metrics are macro-averages over their task groups; `eval_abstention_rate` is the example-count-weighted mean of official task abstention rates
- eval ordering or batching constraint: task-id-first execution is mandatory here; `train.py` sorts by official `task_id`, evaluates one task at a time, and only batches requests within the current task using `--max-concurrent-requests`
- official scorer runtime requirement: the experiment `pyproject.toml` includes the vendored LawBench scorer dependencies used by `train.py` (for example `rouge-chinese`, `jieba`, `cn2an`, `opencc-python-reimplemented`, and `python-Levenshtein`), so a clean `uv sync` is sufficient for the official eval path

### Local deviations from upstream

- model: the maintained local execution baseline is `Qwen/Qwen3-4B-Instruct-2507`, not the original paper backbones such as Qzhou-Law variants or LawLLM-7B.
- data: train uses the public `DISC-Law-SFT` release and eval uses a frozen local JSONL manifest converted from the official raw task files.
- training: the local route is LoRA SFT plus final official eval, with task-id-first eval and bounded train-lineage logging.

## Current results

Status: `placeholder`

No maintained reportable run is checked in yet for this experiment.

When you add one, keep the evidence bundle together: SFT config, final eval config, `eval_lawbench_avg`, wall-clock timing, and the artifact directory path for the corresponding benchmark confirmation run.

## References

- Requirement: `requirements/lawbench-on-mint/README.md`
- Benchmark repo: `https://github.com/open-compass/LawBench`
- Benchmark paper: `https://aclanthology.org/2024.emnlp-main.452/`
- Maintained train source: `https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT`
