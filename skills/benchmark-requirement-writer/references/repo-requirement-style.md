# Repo Requirement Style

Use this reference when writing requirement docs inside this repository.

## Canonical location

- Save benchmark requirement docs under `requirements/<benchmark>-on-mint/README.md`.
- Use lowercase benchmark names in the directory name when reasonable.

Examples:

- `requirements/aime-on-mint/README.md`
- `requirements/finqa-on-mint/README.md`
- `requirements/lawbench-on-mint/README.md`

## Canonical repo assumptions

Match the current repo contract:

- benchmark entrypoint: `uv run train.py --eval-only --eval-data <full_eval_path>`
- train entrypoint: `uv run train.py`
- dry-run validation: `uv run train.py --dry-run --eval-data <smoke_eval_path>`
- automation entrypoint: `autoresearch.sh`
- experiment directories live under `experiments/`
- each experiment should stay self-contained

## Writing pattern

The local requirement docs usually work best with this section order:

1. Goal
2. Benchmark choice / benchmark contract
3. Latest public results summary
4. Reproduction target method
5. Comparison baselines and execution baselines
6. Algorithm route decision
7. Data requirements
8. `train.py` requirements
9. Metrics and acceptance milestones
10. References

Not every document needs the exact same titles, but the structure should stay recognizable.

## Repo-specific expectations to mention

Unless the user asks otherwise, requirement docs in this repo should usually state:

- the target experiment directory, usually `experiments/<benchmark>/`
- that the benchmark path should be frozen early
- that baseline eval should happen before training claims
- that `train.py` must support `--eval-only`
- that `train.py` should support `--dry-run` when possible
- that metrics should be emitted as `METRIC name=value`
- that `autoresearch.sh` is the canonical automation entrypoint
- that phase-one CLI should stay minimal rather than exposing every training hyperparameter

## Target-method and baseline wording

In this repo, requirement docs should separate the reproduced method goal from the runnable cookbook path:

- name the exact `reproduction target method`
- if the user only gives a benchmark, default that target method to the latest benchmark-tested method that is public enough to ground the plan
- name the exact public `comparison baseline` only when it helps contextualize the target method with citable benchmark numbers
- name the exact primary `execution baseline` that will be run first in this repo
- if needed, name the exact secondary `execution baseline`
- explain whether the execution baseline matches the target method backbone or is a smaller/different runnable baseline chosen to chase the same benchmark effect
- explain how the chosen execution baseline affects whether the first algorithm should be `SFT`, `DPO`, or `GRPO`
- if primary and secondary execution baselines imply different first algorithms, say that explicitly
- when algorithm choice is gated by the runnable baseline, put `Reproduction target method` and `Comparison baselines and execution baselines` before `Algorithm route decision`
- in phase one, avoid forcing a generic `--algorithm` switch unless one `train.py` truly hosts multiple stable routes
- if such a future switch is needed, prefer a concrete name such as `--algorithm`

Avoid vague wording like "start with a reasonable baseline" or "reproduce a benchmark" without naming both the target method and the first runnable execution baseline.
