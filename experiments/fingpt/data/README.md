# Data contract

This experiment keeps benchmark artifacts local.

## Canonical Fineval slice

The current runnable experiment contract still uses the official Fineval slice as the canonical benchmark path:

```text
data/fingpt-fineval/
├── train.jsonl
└── test.jsonl
```

Create those files with:

```bash
cd experiments/fingpt
uv sync
uv run python data/download_fingpt_fineval.py
```

The downloader pulls the public Hugging Face dataset `FinGPT/fingpt-fineval` and rewrites each split into local JSONL.
Each row preserves the instruction-following shape used by FinGPT:

- `id`
- `instruction`
- `input`
- `output`

## Sentiment train data

The public sentiment training corpus currently kept locally is:

```text
data/fingpt-sentiment-train/
└── train.jsonl
```

This mirrors `FinGPT/fingpt-sentiment-train` and is the local training corpus when the experiment later needs a public sentiment SFT path.

Create or refresh it with:

```bash
cd experiments/fingpt
uv sync
uv run python data/download_fingpt_sentiment.py
```

## Official sentiment train/eval contract

According to the official FinGPT sentiment README and data notebooks, the sentiment line should be read as a train/eval package rather than as four test-only benchmark names.

| Dataset | Train count | Eval count | Upstream source |
| --- | ---: | ---: | --- |
| `FPB` | `3634` | `1212` | `financial_phrasebank` (`sentences_50agree`) + local split |
| `FiQA-SA` | `938` | `275` | `pauri32/fiqa-2018` + local split |
| `TFNS` | `9543` | `2388` | `zeroshot/twitter-financial-news-sentiment` |
| `NWGI` | `16184` | `4047` | `oliverwang15/news_with_gpt_instructions` |

The often-quoted `3634 / 938 / 9543 / 16184` numbers are the train-side counts that feed into `fingpt-sentiment-train`. The held-out evaluation side is `1212 / 275 / 2388 / 4047`.

## Sentiment benchmarks (official FinGPT notebook alignment)

The local sentiment benchmark mirrors below now follow the upstream dataset choices and split logic shown in the official FinGPT sentiment notebooks and README:

```text
data/benchmarks/sentiment/
├── fpb/
│   ├── train.jsonl
│   └── test.jsonl
├── fiqa-sa/
│   ├── train.jsonl
│   └── test.jsonl
├── tfns/
│   ├── train.jsonl
│   └── test.jsonl
└── nwgi/
    ├── train.jsonl
    └── test.jsonl
```

Create or refresh these files with:

```bash
cd experiments/fingpt
uv sync
uv run python data/download_fingpt_sentiment_benchmarks.py
```

For faster training-time sentiment checks, also build the documented small eval subset:

```bash
cd experiments/fingpt
uv run python data/make_sentiment_train_eval_subset.py
```

Current local interpretation:

- `FPB`
  - upstream: `financial_phrasebank` with config `sentences_50agree`
  - local split rule: `train_test_split(seed=42)` on the full `4846` rows
  - resulting sizes: `train=3634`, `test=1212`
- `FiQA-SA`
  - upstream: `pauri32/fiqa-2018`
  - local split rule: merge `train + validation + test`, bucketize scores into `negative/neutral/positive`, then `train_test_split(test_size=0.226, seed=42)`
  - resulting sizes: `train=938`, `test=275`
- `TFNS`
  - upstream: `zeroshot/twitter-financial-news-sentiment`
  - local rule: upstream `train` stays `train`, upstream `validation` becomes local `test`
  - resulting sizes: `train=9543`, `test=2388`
- `NWGI`
  - upstream: `oliverwang15/news_with_gpt_instructions`
  - local rule: upstream `train/test` copied directly
  - resulting sizes: `train=16184`, `test=4047`

This is the first local version whose sentiment benchmark counts line up with the official FinGPT sentiment README: the often-quoted `3634 / 938 / 9543 / 16184` numbers are the train-side counts, while the corresponding held-out benchmark sizes are `1212 / 275 / 2388 / 4047`.

## Training-time sentiment eval subset

For cheap periodic eval during future sentiment-side training runs, this experiment also documents a small stratified subset under:

```text
data/benchmarks/sentiment/train-eval-160/
├── manifest.json
├── all/
│   └── test.jsonl
├── fpb/
│   └── test.jsonl
├── fiqa-sa/
│   └── test.jsonl
├── tfns/
│   └── test.jsonl
└── nwgi/
    └── test.jsonl
```

Generate it with:

```bash
cd experiments/fingpt
uv run python data/make_sentiment_train_eval_subset.py
```

Current local interpretation:

- the subset builder preserves the benchmark mix proportionally across `FPB`, `FiQA-SA`, `TFNS`, and `NWGI`
- within each benchmark, rows are sampled in a label-stratified way so every non-empty sentiment label still appears
- `all/test.jsonl` is the documented default eval file for training-time spot checks only; final reportable eval still uses the full held-out benchmark bundle

## Smoke fallback

Before the official Fineval dataset is downloaded, `train.py` can still fall back to:

- `data/smoke_train.jsonl`
- `data/smoke_eval.jsonl`

These are tiny hand-written examples for local validation only.
They should never be treated as the reproduced benchmark.

Canonical explicit eval-data commands:

- `uv run train.py --dry-run --task-type fineval --eval-data smoke:data/smoke_eval.jsonl`
- `uv run train.py --eval-only --task-type fineval --eval-data fineval:data/fingpt-fineval/test.jsonl`
- `uv run train.py --eval-only --task-type sentiment --eval-data <full held-out sentiment bundle>` where the bundle includes `FPB`, `FiQA-SA`, `TFNS`, and `NWGI`
