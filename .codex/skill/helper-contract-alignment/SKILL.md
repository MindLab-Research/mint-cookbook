---
name: helper-contract-alignment
description: |
  Reconcile same-named shared helpers across experiments, scaffold templates,
  profiles, and repo docs without silently changing the contract under the same
  name.

  Use when repo-wide helper names drift, when a helper is promoted into
  `scaffolds/single_file_experiment/naming.md`, or when experiment/template/
  profile/skill updates must stay synchronized.

  Core behavior: scan repeated helper names, classify shared-contract drift vs
  task-specific differences, choose a canonical implementation, patch live code
  first, then sync templates/docs/skills, and verify the aligned helpers with
  AST/tests/hash checks.
---

# Helper Contract Alignment

Use this skill when the task is repo-wide reconciliation of shared helper
contracts, not when working on only one experiment's local logic.

## When this skill applies

- The user asks to align same-named functions across multiple experiments
- A helper is being promoted into `scaffolds/single_file_experiment/naming.md`
- `scaffolds/` and live experiments have drifted on helper names or signatures
- The user says "same function names should stay consistent" or asks whether a
  helper is truly shared or should become a wrapper
- A previous refactor updated one experiment but not the template, profiles, or
  skills

Do not use this skill for:

- Debugging one experiment's benchmark logic
- Task-specific grader or prompt changes
- Pure API troubleshooting inside one experiment

For those cases, use the experiment-local workflow, `$mint-api`,
`$new-experiment`, or `$new-experiment-plus` as appropriate.

## Hard rules

- Same name implies the same semantic contract.
- If an experiment needs extra behavior, keep the shared helper stable and add
  a differently named wrapper.
- Shared helper ownership order is:
  1. `scaffolds/single_file_experiment/naming.md`
  2. `scaffolds/profiles/*.md`
  3. `scaffolds/single_file_experiment/train.py.tpl`
  4. live experiments
- Modify live code first, then sync template/docs/skills in the same phase.
- Do not silently leave scaffold docs or skills behind after changing a shared
  contract.

## Workflow

1. Define the scope and exclusions first.
   - Example: exclude specific experiments if the user says they are out of scope.
2. Run the bundled scan script to find repeated top-level helper names.
3. Classify each drifted helper:
   - shared infra/helper drift
   - SFT/GRPO internal contract drift
   - task-specific helper that should stay local
4. Choose the canonical implementation before editing.
   - If two live experiments plus the template already match, usually align the
     outlier to that shape.
5. Patch live experiments first.
6. Validate immediately:
   - AST parse touched files
   - targeted tests or small self-checks
   - hash compare helpers that should now be strictly identical
7. Sync `train.py.tpl`, `naming.md`, profiles, and skills when the shared
   contract changed materially.
8. Re-scan and list any remaining intentional differences.

## Choosing canonical behavior

- Prefer the implementation already shared by the majority of live experiments
  plus `train.py.tpl`.
- Keep task-specific adapters out of the shared-helper set even if names
  happen to match.
- If a GRPO or SFT experiment needs extra orchestration around a shared helper,
  split it into:
  - shared helper: canonical name
  - local orchestration: wrapper with a new name

## Validation checklist

For each alignment phase:

- AST parse every touched `.py` file
- Parse `train.py.tpl` if it remains valid Python
- Run targeted experiment tests when available
- Use helper hashes to confirm strict identity where intended
- Check that docs/template/skills were updated in the same commit when the
  contract changed

## Use bundled resources

- `scripts/scan_helper_drift.py`
  - scan repeated top-level helpers and compare hashes
  - supports include/exclude patterns and helper-name filters
  - use `--ignore ...` for known intentional variants so the scan stays focused
- `references/alignment-rules.md`
  - classification rules, owner map, and a starter whitelist of likely shared
    helpers
- `references/single-file-experiment-review-checklist.md`
  - repo-specific review checklist for scaffold drift, artifact layout, resume
    semantics, and the three live smoke flows across single-file experiments

## Typical commands

```bash
python .codex/skill/helper-contract-alignment/scripts/scan_helper_drift.py \
  --exclude "experiments/finqa/*"
```

```bash
python .codex/skill/helper-contract-alignment/scripts/scan_helper_drift.py \
  --only create_training_client save_training_state save_sampler_checkpoint
```

```bash
python .codex/skill/helper-contract-alignment/scripts/scan_helper_drift.py \
  --ignore build_eval_messages grade_assistant_text compute_eval_metrics \
           normalize_train_rows normalize_eval_rows write_outputs run_train \
           build_supervised_batch_trace_record create_service_client \
           parse_args run_dry_run run_eval main_async
```
