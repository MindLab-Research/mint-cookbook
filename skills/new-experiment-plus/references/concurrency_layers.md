# Concurrency Layers

Use this file when the user says training is slow and the experiment already has a meaningful baseline.

## Core Reminder

Do not collapse every throughput problem into "need more concurrency".
A common failure mode is:

- rollout collection is already async enough
- but each step still waits for the full nominal batch
- so no training work starts until the slowest tail requests finish

In that case, raising `max_concurrent_requests` alone may not help much.

## Layer 1: Sampling Shape

Questions to answer first:

- Does one request return `num_samples=group_size`?
- Or does the code form each group from repeated `num_samples=1` requests?

This changes how you reason about latency, ordering, retries, and replacement sampling.

## Layer 2: In-Flight Fan-Out

Typical knob:

- `max_concurrent_requests`

This controls how many requests are in flight at once. It is useful, but it does not remove a strong step barrier by itself.

## Layer 3: In-Step Overlap

Typical knob:

- `stream_minibatches_per_step`

Use this when training can start before the full nominal batch is ready. The usual pattern is:

1. collect rollouts
2. start `forward_backward` as soon as one minibatch is ready
3. keep collecting later groups in parallel
4. keep a single `optim_step` at the end of the step

## Layer 4: Tail Control

Typical knobs:

- `min_accepted_groups`
- `min_started_minibatches`
- `tail_grace_seconds`

Use these when the final slow or invalid groups dominate wall-clock time without materially changing the useful training signal.

## Layer 5: Train/Eval Overlap

Questions to answer:

- Does periodic eval block training?
- Are evals skipped or queued?
- Does eval use the correct saved sampler snapshot?

For long runs, checkpoint-aware background eval with queueing is often better than either full blocking or silent skipping.

## Preferred Optimization Order

1. Confirm the benchmark and artifact contract first.
2. Measure where wall-clock time is actually going.
3. Add same-step streaming minibatches.
4. Add tail early-stop if the final groups dominate latency.
5. Only then push harder on raw async fan-out or queue/worker complexity.

## What To Log

When changing throughput behavior, log enough to explain the effect:

- nominal vs effective batch shape
- accepted groups
- trained groups
- dropped groups
- early-stop reason
- step time
- samples per second or equivalent throughput number
- rollout-wave or replacement-wave counters when relevant
