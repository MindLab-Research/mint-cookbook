# Data bootstrap

If `data/train/full.jsonl` is missing or you want to refresh the split local artifacts from upstream, rebuild them with:

```bash
cd experiments/dapo-aime
python data/download_and_prepare.py
```

Default download source:

- China mirror: `https://hf-mirror.com`

What the helper does:

- downloads the public `BytedTsinghua-SIA/DAPO-Math-17k` parquet from the Hugging Face China mirror
- writes or reuses `data/train/raw_dapo_math_17k.parquet`
- deduplicates DAPO rows by `extra_info.index`
- extracts the math problem from the original prompt template
- writes the default local training file expected by this experiment to `data/train/full.jsonl`
- writes the train smoke slice to `data/train/smoke.jsonl`
- downloads or reuses the public `Maxwell-Jia/AIME_2024` train split jsonl from the Hugging Face China mirror
- writes or reuses `data/eval/raw_aime_2024.jsonl`
- writes `data/eval/aime2024.jsonl`
- downloads or reuses the public `MathArena/aime_2025` parquet from the Hugging Face China mirror
- writes or reuses `data/eval/raw_aime_2025.parquet`
- converts the MathArena rows into the local eval schema and writes `data/eval/aime2025.jsonl`
- downloads or reuses the public `MathArena/aime_2026` parquet from the Hugging Face China mirror
- writes or reuses `data/eval/raw_aime_2026.parquet`
- converts the MathArena rows into the local eval schema and writes `data/eval/aime2026.jsonl`
- writes `data/eval/smoke.jsonl` from one row each from AIME 2024, 2025, and 2026

Mirror download URLs used by the helper:

- `https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet`
- `https://hf-mirror.com/datasets/Maxwell-Jia/AIME_2024/resolve/main/train.jsonl`
- `https://hf-mirror.com/datasets/MathArena/aime_2025/resolve/main/data/train-00000-of-00001.parquet`
- `https://hf-mirror.com/datasets/MathArena/aime_2026/resolve/main/data/train-00000-of-00001.parquet`

Upstream dataset pages:

- `https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k`
- `https://hf-mirror.com/datasets/Maxwell-Jia/AIME_2024`
- `https://hf-mirror.com/datasets/MathArena/aime_2025`
- `https://hf-mirror.com/datasets/MathArena/aime_2026`

Useful options:

```bash
# use a different mirror or endpoint
python data/download_and_prepare.py --mirror-base https://hf-mirror.com

# only reuse local raw files; do not touch the network
python data/download_and_prepare.py --local-only

# refresh raw files in a separate directory
python data/download_and_prepare.py --data-dir /tmp/dapo-aime-data --force-download
```

Expected local files after bootstrapping:

- `data/train/raw_dapo_math_17k.parquet`
- `data/train/full.jsonl`
- `data/train/smoke.jsonl`
- `data/eval/raw_aime_2024.jsonl`
- `data/eval/raw_aime_2025.parquet`
- `data/eval/raw_aime_2026.parquet`
- `data/eval/aime2024.jsonl`
- `data/eval/aime2025.jsonl`
- `data/eval/aime2026.jsonl`
- `data/eval/smoke.jsonl`

The training file rows look like:

```json
{"id": "...", "question": "...", "answer": "34", "source": "BytedTsinghua-SIA/DAPO-Math-17k"}
```

The evaluation files keep the `Problem` / `Answer` row shape used by `train.py`:

```json
{"ID": "2024-I-1", "Problem": "...", "Solution": "...", "Answer": 204}
```

Notes:

- Use `--eval-data data/eval/smoke.jsonl` for local `--dry-run` validation and `--eval-data data/eval/aime2024.jsonl` for held-out eval or training final eval.
- `--eval-data` also accepts comma-separated entries such as `data/eval/aime2024.jsonl,data/eval/aime2025.jsonl,data/eval/aime2026.jsonl` or explicit `name:path` pairs.
- When multiple eval manifests are named, final eval writes one full artifact bundle per dataset under `eval/<name>/...`, while training-time periodic eval uses one stable random third from each dataset to build a fixed mixed eval set.
- `data/eval/aime2024.jsonl`, `data/eval/aime2025.jsonl`, and `data/eval/aime2026.jsonl` are the explicit year-named eval manifests.
- `data/eval/smoke.jsonl` and `data/train/smoke.jsonl` are kept as tiny repo-local validation artifacts so explicit smoke validation works before the full train artifact is materialized.
- `data/train/full.jsonl` remains the local materialized train artifact for real GRPO runs.
- Keep root `data/` for docs and helper scripts only. Raw snapshots and JSONL artifacts belong under `data/train/` or `data/eval/`, not as top-level aliases.
- If raw files already exist, the helper reuses them unless you pass `--force-download`.
- If a raw file path is a symlink, the helper reuses it instead of overwriting the shared target.
- The helper is idempotent for local materialization.
- `data/eval/smoke.jsonl` is rebuilt from the first row of each year-named AIME eval manifest, while `data/train/smoke.jsonl` is rebuilt from the first row of `data/train/full.jsonl`.
