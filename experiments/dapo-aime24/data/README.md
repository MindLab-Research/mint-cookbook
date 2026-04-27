# Data bootstrap

If `data/train.jsonl` or `data/eval.jsonl` is missing, rebuild them with:

```bash
cd experiments/dapo-aime24
python data/download_and_prepare.py
```

Default download source:

- China mirror: `https://hf-mirror.com`

What the helper does:

- downloads the public `BytedTsinghua-SIA/DAPO-Math-17k` parquet from the Hugging Face China mirror
- writes or reuses `data/raw_dapo_math_17k.parquet`
- deduplicates DAPO rows by `extra_info.index`
- extracts the math problem from the original prompt template
- writes the local training file expected by this experiment to `data/train.jsonl`
- downloads or reuses the public `Maxwell-Jia/AIME_2024` train split jsonl from the Hugging Face China mirror
- writes the frozen local evaluation file to `data/eval.jsonl`

Mirror download URLs used by the helper:

- `https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet`
- `https://hf-mirror.com/datasets/Maxwell-Jia/AIME_2024/resolve/main/train.jsonl`

Upstream dataset pages:

- `https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k`
- `https://hf-mirror.com/datasets/Maxwell-Jia/AIME_2024`

Useful options:

```bash
# use a different mirror or endpoint
python data/download_and_prepare.py --mirror-base https://hf-mirror.com

# only reuse local raw files; do not touch the network
python data/download_and_prepare.py --local-only

# refresh raw files in a separate directory
python data/download_and_prepare.py --data-dir /tmp/dapo-aime24-data --force-download
```

Expected local files after bootstrapping:

- `data/raw_dapo_math_17k.parquet`
- `data/train.jsonl`
- `data/eval.jsonl`

The training file rows look like:

```json
{"id": "...", "question": "...", "answer": "34", "source": "BytedTsinghua-SIA/DAPO-Math-17k"}
```

The evaluation file keeps the original AIME row shape used by `train.py`:

```json
{"ID": "2024-I-1", "Problem": "...", "Solution": "...", "Answer": 204}
```

Notes:

- `train.py` reads `data/train.jsonl` and `data/eval.jsonl` by default.
- If raw files already exist, the helper reuses them unless you pass `--force-download`.
- If a raw file path is a symlink, the helper reuses it instead of overwriting the shared target.
- The helper is idempotent for local materialization.
