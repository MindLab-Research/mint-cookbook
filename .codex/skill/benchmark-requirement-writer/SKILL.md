---
name: benchmark-requirement-writer
description: Write benchmark reproduction requirement documents from a benchmark name plus an optional target method. Use when the user wants to reproduce, benchmark, evaluate, or plan work on a public benchmark and needs a requirement doc, latest literature/results summary, a reproduced method target, comparison baselines, reference reproduction methods, or an algorithm choice between SFT, DPO, and GRPO. If the user does not provide a method, first research the latest academic methods tested on that benchmark, choose a reproduction target, and then pick a practical execution baseline that can be pushed toward that target effect.
argument-hint: "[benchmark-name] [optional-target-method]"
---

# Benchmark Requirement Writer

Write a requirement document for a benchmark reproduction project.

## Repo-default interpretation

In this repo, a benchmark request usually means "find the latest benchmark-tested method, treat that as the target method, and then choose a practical cookbook execution baseline that can be run locally and pushed toward that target effect."

That means:
- do not collapse `target method` and `execution baseline` into the same thing unless the user explicitly wants that
- it is valid for the target method to come from one paper backbone while the runnable cookbook baseline uses another backbone such as `Qwen3-4B`
- when this happens, write the requirement so the target method defines the desired benchmark effect, while the execution baseline defines the first runnable eval/train path
- if exact same-backbone reproduction is impossible or not the repo goal, say that explicitly instead of pretending the execution baseline is the reproduced paper itself

## Step 1: Understand the request

Extract these fields from the user request before drafting:
- **Benchmark**: the public benchmark or dataset to reproduce
- **Target method**: optional named paper or method to reproduce
- **Target models / execution baselines**: optional preferred runnable baselines such as Qwen or Tinker-supported models
- **Comparison baselines**: optional named public baselines to compare against
- **Reference methods**: optional open-source or partially open methods whose training route or data should be referenced
- **Repo constraints**: local repo contract, required entrypoints, and experiment layout

If the user gives a benchmark and no method, first research the latest benchmark-tested methods before writing.
If the user gives target models but no public method target, plan to separate a reproduced method target, public comparison baselines, and the practical execution baseline.
If the user gives both a benchmark and a preferred runnable model such as `Qwen3-4B`, keep that runnable model as the likely `execution baseline` unless the user says they want exact same-backbone reproduction.

## Step 2: Research benchmark-tested methods first

Browse before drafting whenever benchmark results, papers, leaderboards, or model availability may have changed.

Prefer primary sources:
- official benchmark repo or site
- official paper
- official leaderboard
- ACL / arXiv / OpenReview papers that report benchmark-wide results

Research rules:
- use absolute dates when discussing latest results
- distinguish full-benchmark results from subset-only or derivative results
- separately track what is public for `results`, `models`, `training code`, and `training data`
- if the user only provides a benchmark, identify the latest benchmark-tested methods before choosing comparison baselines
- if the user's preferred model has no public result, say that clearly and recommend the nearest citable comparison point
- if a method is only partially open-source, state exactly which parts are available
- if you searched for benchmark-wide `DPO` or `GRPO` results and did not find any, state that as a search finding rather than an absolute claim

## Step 3: Select the reproduced method first

Separate four roles that are easy to confuse:
- `reproduction target method`: the benchmark-tested paper or method you are actually trying to reproduce
- `comparison baseline`: the public, citable comparison line
- `execution baseline`: the runnable model you actually plan to evaluate and train from
- `reference method`: the public method, data release, or training recipe you plan to imitate when constructing the training path

Selection rules:
- the reproduced method should set the goal statement near the top of the requirement; comparison baselines and execution baselines exist to contextualize and operationalize that goal, not replace it
- if the user already names a target method, keep it unless research shows it is non-public, non-comparable, or a poor fit for the intended benchmark setup
- if the user only names a benchmark, default to the latest benchmark-tested method that is public enough to ground a reproduction plan
- if the repo goal is cookbook-style reproduction, allow the chosen execution baseline to differ from the target method's original backbone, but make the gap explicit
- if the user names target models but not a public method target, choose one reproduced method target plus one practical execution baseline
- when the user needs training guidance, also name at least one `reference method` and explain whether it is useful for `data`, `training recipe`, `engineering`, or `full reproduction`
- add comparison baselines only when they help situate the reproduced method against public benchmark numbers
- add a secondary execution baseline only when the user clearly cares about two scales or two model families
- state early which method is being reproduced and whether other named items are target methods, comparison baselines, execution baselines, or reference methods
- explain why the chosen reproduced method beats nearby alternatives for this project

## Step 4: Route the algorithm from the execution baseline

Do not lock in `SFT`, `DPO`, or `GRPO` before execution baseline profiling. The algorithm recommendation should be driven by the first runnable baseline and the size of its gap to the target method, not by the target paper name alone.

Use this routing logic:
- choose `SFT` first when the gap is mainly domain knowledge, output format, task framing, or instruction following
- choose `DPO` first when the likely gain comes from preference shaping, refusal/style tradeoffs, or chosen-vs-rejected supervision
- choose `GRPO` first only when the baseline is already fairly strong and the benchmark has a reward or verifier that can be defined clearly and audited

If there are multiple execution baselines, allow different first-route recommendations for different baselines.
Read `references/algorithm-routing.md` when the main uncertainty is whether the project should start with `SFT`, `DPO`, or `GRPO`.

## Step 5: Write the requirement in repo style

If you are writing inside this repo:
- save to `requirements/<benchmark>-on-mint/README.md` unless the user asks for another location
- align with the local MinT experiment contract instead of inventing a new format
- use `uv run train.py --eval-only` as the benchmark entrypoint
- use `uv run train.py` as the train entrypoint
- use `uv run train.py --dry-run` as the dry-run validation entrypoint when possible
- use `autoresearch.sh` as the automation entrypoint

Read these local references when writing into this repo:
- `README.md`
- `experiments/README.md`
- `requirements/`
- `references/repo-requirement-style.md`

## Output pattern

When the user only provides a benchmark, the final requirement should normally make this chain explicit near the top:
1. `benchmark`: what is being evaluated
2. `target method`: which latest benchmark-tested method defines the desired effect
3. `comparison baselines`: which public numbers provide citable context
4. `execution baseline`: which runnable model will be evaluated first in this repo
5. `training route`: how that execution baseline will be pushed toward the target method effect

A short one-line framing often works well, for example:
- "Target method is `X`; first runnable execution baseline is `Y`; the cookbook goal is to push `Y` toward `X` on benchmark `B`."

Default section order for the requirement doc:
1. Goal
2. Benchmark definition and contract
3. Latest public results summary
4. Reproduction target method
5. Comparison baselines and execution baselines
6. Algorithm route decision
7. Reference reproduction methods
8. Data and leakage constraints
9. Metrics and acceptance milestones
10. Sources

When algorithm choice depends strongly on method choice, keep `Reproduction target method` and `Comparison baselines and execution baselines` before `Algorithm route decision`.
Do not recommend a generic `--algorithm` CLI switch in phase one unless one `train.py` really needs to host multiple stable training routes.

## Step 6: Validate the draft

Before you finish, check that the requirement doc:
- names the exact benchmark being targeted
- names the exact reproduced method target and labels other named entities as target methods, comparison baselines, execution baselines, or reference methods
- makes clear whether the execution baseline matches the target method backbone or is a smaller/different cookbook baseline chosen to chase the same benchmark effect
- explains why the chosen reproduced method beats nearby alternatives for this project
- states which public methods are worth imitating for training and what exactly is open-source about them
- states which algorithm should be tried first for each execution baseline and why
- labels recommendations and inferences clearly instead of presenting them as quoted facts
- links the sources used in the final requirement
- matches the local repo contract if written inside a repo

## Local references

Read only what the task needs:
- `references/algorithm-routing.md` - quick routing guide for `SFT`, `DPO`, and `GRPO`
- `references/repo-requirement-style.md` - local requirement format and repo contract
