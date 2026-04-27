# Data layout

This experiment keeps one canonical local pair format:

- `data/train.jsonl` - preference pairs used for DPO training
- `data/eval.jsonl` - held-out preference pairs used for benchmark eval
- `data/sources.yaml` - provenance, filtering, overlap, and migration notes

The default JSONL paths should contain tiny local scaffold rows or real
preference data so `--dry-run` and unit tests have a concrete local contract to
validate. JSONL files are gitignored in this repo, so create or sync them
locally before running the experiment. Scaffold rows are not reportable data.

Future raw/processed mirrors from legacy `chat_dpo` may be stored here, for
example:

- `data/raw/`
- `data/processed/`

But the default runtime paths remain `data/train.jsonl` and `data/eval.jsonl`.
