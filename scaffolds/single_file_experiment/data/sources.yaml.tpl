experiment: {{EXPERIMENT_NAME}}
eval_profile: {{EVAL_STYLE}}
train_profile: none
eval_split_rule: pending
train_split_rule: pending
canonical_train_artifact: pending
canonical_eval_artifact: data/eval/full.jsonl
canonical_eval_smoke_artifact: data/eval/smoke.jsonl
canonical_train_smoke_artifact: pending
canonical_eval_raw_dir: pending
canonical_train_raw_dir: pending
eval_downloader: pending
eval_builder: pending
train_downloader: pending
train_builder: pending
notes:
  - Every new experiment starts eval-first; update train_profile and canonical_train_artifact only after the eval contract is stable.
  - Prefer split dataset management under data/eval/ and data/train/ for single-dataset experiments; multi-dataset experiments may instead group artifacts by dataset family under data/ when that matches the upstream sources better.
  - Prefer preserving the local reconstruction pipeline: raw snapshots plus download or snapshot scripts and build or adjustment scripts kept under data/, next to the artifacts they manage.
  - When practical, materialize smoke first and full second on both sides.
  - Replace artifact names above if the benchmark keeps baseline-native filenames.
  - Record merge rules, local adapters, and evaluation split provenance here.
sources:
  - name: replace-me
    role: eval
    path: replace-me
    format: jsonl
    raw_snapshot_dir: pending
    downloader: pending
    builder: pending
  - name: optional-train-source
    role: train
    path: pending
    format: jsonl
    raw_snapshot_dir: pending
    downloader: pending
    builder: pending
