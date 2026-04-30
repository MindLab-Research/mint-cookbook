# Data layout

This experiment keeps one canonical local pair format:

- `data/train/full.jsonl` - default preference-pair file used for DPO training
- `data/train/smoke.jsonl` - tiny train-side smoke slice or placeholder pairs
- `data/eval/full.jsonl` - default held-out preference-pair file used for benchmark eval
- `data/eval/smoke.jsonl` - tiny eval-side smoke slice or placeholder pairs
- `data/sources.yaml` - provenance, filtering, overlap, and migration notes

Keep root `data/` for docs and side directories only. Do not keep top-level
`*.jsonl` aliases there; train-side artifacts belong under `data/train/` and
eval-side artifacts belong under `data/eval/`.

The split files under `data/train/` and `data/eval/` are kept as tiny local
validation artifacts so explicit smoke/full commands and unit tests have a
concrete contract immediately after checkout. Use `--eval-data
data/eval/smoke.jsonl` for `--dry-run` validation and `--eval-data
data/eval/full.jsonl` for held-out eval or training final eval. Scaffold rows
are not reportable data; replace the checked-in `full.jsonl` placeholders with
local real preference data before treating any run as a real result.

Future raw or processed mirrors from legacy `chat_dpo` may be stored under the
matching side, for example:

- `data/train/raw/`
- `data/train/processed/`
- `data/eval/raw/`
- `data/eval/processed/`

But the default runtime paths remain `data/train/full.jsonl` and `data/eval/full.jsonl`.
