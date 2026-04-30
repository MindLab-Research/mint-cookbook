---
name: mint-api
description: |
  MinT SDK API reference and production training patterns.

  Use when writing or modifying training scripts that call `import mint`:
  correct API signatures, loss functions, data types, async concurrency patterns,
  checkpoint management, and debugging.

  Covers: SFT (cross_entropy), RL/GRPO (Tinker-aligned `importance_sampling`,
  with `ppo` called out separately when needed), custom loss
  (forward_backward_custom for DPO), async concurrent sampling with
  sliding-window concurrency, session recovery, and framework migration (verl/TRL → MinT).
---

# MinT SDK Quick Reference

Use this skill when writing or reviewing code that uses `import mint`.

## When this skill applies

- Writing a new `train.py` for an experiment
- Debugging mint API calls (wrong argument names, missing `.result()`, async await-shape mismatches, etc.)
- Choosing between loss functions (`cross_entropy`, `importance_sampling`, and when `ppo` is an explicit experiment choice rather than the default Tinker-aligned path)
- Preparing `types.Datum` objects from raw data
- Setting up SFT, RL (GRPO), or custom loss (DPO/Bradley-Terry) training loops
- Writing async training loops with concurrent sampling
- Checkpoint save/load/resume workflows
- Migrating from verl, TRL, or OpenRLHF to MinT

## Cheatsheet sections

The file `mint_api_cheatsheet.md` in this directory covers:

1. **Client Creation** — ServiceClient, TrainingClient, tokenizer
2. **Data Types** — Datum, ModelInput, SamplingParams, AdamParams
3. **SFT Training** — cross_entropy loss, data prep, training loop
4. **RL Training (GRPO / Rollout Groups)** — rollout collection → reward → advantage → `importance_sampling`
5. **Built-in MinT / Tinker Losses** — `cross_entropy`, Tinker-aligned `importance_sampling`, and `ppo` caveats
6. **Custom Loss Functions** — `forward_backward_custom` for DPO/Bradley-Terry
7. **Rollout / Sampling Clients** — rollout requests, prompt logprobs, create_sampling_client
8. **Checkpoints** — save_state, save_weights_for_sampler, load_state, download
9. **Async Rollout / Training Patterns** — await-shape caveats, overlap, sliding-window concurrency, sampler holder, pitfalls
10. **Hyperparameter Guidance** — LoRA LR scaling, rank selection, RL tuning
11. **Supported Models** — Qwen3 series (check `get_server_capabilities()` for latest)
12. **Common Patterns** — env loading, preflight check, METRIC output
13. **Framework Migration Reference** — verl → MinT, TRL → MinT mapping tables

## Current repo coverage

- Live smoke coverage in `tests/` currently includes sync SFT, async SFT API shape, checkpoint resume through fresh `create_lora_training_client(...)` plus `load_state_with_optimizer(...)`, and a combined loss smoke in `tests/loss_smoke.py` that exercises `forward_backward_custom` plus minimal `importance_sampling` and `ppo` RL paths

## Resume rule

- On the current MinT endpoint, true resume should use a fresh `create_lora_training_client(...)` and then `load_state_with_optimizer(...)`
- Do not recommend `create_training_client_from_state(...)` as the default path here; the helper depends on `/api/v1/weights_info`, which is not available on this endpoint
- Do not treat live-client `load_state(...)` as a safe resume recipe for Megatron LoRA training; local smoke runs hit CUDA illegal memory access on that path
- Treat `load_state(...)` as weight reload only unless the user explicitly wants weights without optimizer state

## Repo-standard async helper set

When you need reusable async MinT sampling in this repo, prefer copying one small, consistent helper set together:

- `resolve_api_result_async(...)` — normalize `result_async()`, awaitables, `asyncio.Future`, or sync `.result()` return shapes
- `sample_one_async(...)` — prefer `sample_async`, otherwise submit sync sampling off the event loop and then resolve the returned shape
- `sample_many_async(...)` — bounded sliding-window helper for prompt-group rollout fan-out and other throughput-oriented sampling pools
- `sample_many_with_semaphore_async(...)` — cookbook-style lightweight eval helper using `Semaphore`, `create_task(...)`, and `gather(...)`
- `asyncio.Queue` + rollout workers + `training_loop()` — cookbook-style train engine when sampling and optimization should overlap
- `main()` / `main_async()` — keep a familiar sync script entrypoint, but run async-heavy train/eval flows inside one top-level event loop

Copy these helpers as a group when possible. That keeps starter scripts readable and avoids subtle mismatches between experiments.

## Tinker / verl naming alignment

When writing examples in this skill, keep the naming split explicit:

- Prefer Tinker / verl algorithm names in prose and local variables: `group_size`, `groups_per_batch`, `rollout`, `trajectory`, `advantages`, `num_rollout_workers`
- Keep MinT SDK field names only where the API requires them, for example `loss_fn="importance_sampling"` and `loss_fn_inputs["logprobs"]`
- Treat `loss_fn_inputs["logprobs"]` as the behavior-policy / sampling logprobs field, i.e. the same concept verl usually calls `old_log_probs`
- Avoid overloaded RL names like `batch_size`, `sample`, or `batch_requests` in GRPO examples when `groups_per_batch`, `trajectory`, and `rollout_requests` are clearer
- Default GRPO examples to `importance_sampling`, because that is the closest Tinker-validated RL path; mention `ppo` only when the experiment explicitly uses PPO-style clipping

## Async / parallel training guidance

- When training is slow, first check whether the real bottleneck is step-level synchronization rather than raw concurrency. A common failure mode is that rollout collection is already async, but each train step still waits for the full nominal batch to finish sampling before any training work starts.
- Prefer throughput fixes in this order: same-step streaming minibatches first, tail-truncation / early-stop second, and more aggressive async or off-policy designs only after those simpler on-policy improvements are clearly insufficient.
- Async knobs should be documented in terms of the actual engine they drive: train-side rollout workers or in-flight sampling pools, and eval-side parallel sampling tasks. Keep algorithm semantics named separately in `group_size` and `groups_per_batch`.
- For cookbook-style train engines, let prompt-level rollout workers push sampled results into an `asyncio.Queue`, then let the trainer consume minibatches from that queue while later rollouts continue sampling.
- When a repo's RL rollout path already samples with repeated `num_samples=1` requests, do not casually rewrite docs or helper guidance to claim `one prompt -> one sample_async(..., num_samples=group_size)` unless you have verified the actual implementation. In this repo's current GRPO path, each prompt-group is formed from repeated single-trajectory rollout requests, which matches the public `tinker-cookbook` token-completer pattern.
- For lightweight eval, prefer `Semaphore(N)` + `create_task(...)` + `gather(...)` over more elaborate queue machinery.
- If replacement sampling creates a long tail on the last few prompt slots, prefer same-step streaming minibatches before reaching for more aggressive async/off-policy designs. Concretely: let accepted groups start `forward_backward_async(...)` as soon as a minibatch is ready, keep a single `optim_step` at the end of the step, and treat this as an on-policy throughput optimization rather than an algorithm change.
- When the real bottleneck is the tail of the final replacement rollout wave, consider allowing the nominal batch to end early instead of forcing every step to hit the full `groups_per_batch` target. A practical pattern is to require a minimum number of accepted groups and started minibatches, then apply a tail timeout (`tail_grace_seconds`) and end the step while logging how many groups were accepted, trained, and dropped.
- Preserve request order. Submit multiple requests concurrently, but store results by request index so reward assignment, logging, and majority-vote evaluation stay deterministic.
- Keep the concurrency contract explicit. If the train engine overlaps rollout sampling with optimization, document how `--groups-per-batch` shapes the nominal algorithm batch, how minibatch streaming starts training within the same step, and how separate knobs such as `--max-concurrent-requests`, `--stream-minibatches-per-step`, `--min-accepted-groups`, `--min-started-minibatches`, and `--tail-grace-seconds` affect throughput without silently changing the core RL objective.
- For RL, parallelize across prompt groups in the step, not inside one trajectory group's reward computation unless the experiment explicitly needs that extra complexity.
- For eval, keep the postprocessing identical to the sync path: gather sampled results concurrently, then run the usual scoring/logging path in request order so `predictions.jsonl`, debug logs, and majority-vote rules stay deterministic.
- If the SDK exposes `sample_async`, use it directly. If the fallback path returns a sync Future, an awaitable, or an object with `result_async()`, resolve that shape after moving the blocking submit call off the event loop thread with `asyncio.to_thread(...)`, then reuse that one-prompt helper inside either the train queue workers or the eval gather helper.
- If eval decode / grading is heavy, keep the concurrency helper simple and do the expensive postprocessing after `gather(...)` returns, or offload that postprocessing explicitly.
- When you overlap `forward_backward` and `optim_step`, do it only after the team has verified the installed SDK shape. Keep starter baselines readable and prefer simple rollout collection → score → train loops unless overlap yields clear benefit.
- Log the concurrency knobs in artifacts and `METRIC` output or run metadata so later comparisons do not confuse throughput changes with algorithm changes.
- For tail-truncated RL steps, log both nominal and effective batch shape: `accepted_groups`, `trained_groups`, `dropped_groups`, and `early_stop`/`early_stop_reason`. This avoids misreading throughput improvements as if they came from training on the same amount of data.
- Dry runs should still validate prompt construction and dataset shape without MinT credentials; async code should degrade cleanly or stay bypassed during `--dry-run`.


### Concrete async rollout example

Use this pattern when you want async concurrency for RL rollout collection or another train-side sampling pool,
but you still want deterministic result ordering and a fixed in-flight cap.

```python
import asyncio

import mint
from mint import types


async def resolve_api_result_async(maybe_result):
    if hasattr(maybe_result, "result_async") and callable(maybe_result.result_async):
        return await maybe_result.result_async()
    if asyncio.isfuture(maybe_result):
        return await maybe_result
    if hasattr(maybe_result, "__await__"):
        return await maybe_result
    if hasattr(maybe_result, "result") and callable(maybe_result.result):
        return await asyncio.to_thread(maybe_result.result)
    return maybe_result


async def sample_one_async(sampler, prompt_tokens, num_samples, sampling_params):
    prompt = types.ModelInput.from_ints(tokens=prompt_tokens)
    sample_async = getattr(sampler, "sample_async", None)
    if callable(sample_async):
        return await sample_async(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )

    def _submit_sample():
        return sampler.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )

    submitted_result = await asyncio.to_thread(_submit_sample)
    return await resolve_api_result_async(submitted_result)


async def sample_many_async(sampler, sampling_requests, max_in_flight):
    if not sampling_requests:
        return []

    max_in_flight = max(max_in_flight, 1)
    results_by_index = [None] * len(sampling_requests)
    pending_tasks = {}
    next_submit_index = 0

    def submit_next_request():
        nonlocal next_submit_index
        if next_submit_index >= len(sampling_requests):
            return
        index = next_submit_index
        prompt_tokens, num_samples, sampling_params = sampling_requests[index]
        pending_tasks[
            asyncio.create_task(
                sample_one_async(sampler, prompt_tokens, num_samples, sampling_params)
            )
        ] = index
        next_submit_index += 1

    for _ in range(min(max_in_flight, len(sampling_requests))):
        submit_next_request()

    try:
        while pending_tasks:
            done_tasks, _ = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done_tasks:
                index = pending_tasks.pop(task)
                results_by_index[index] = task.result()
                submit_next_request()
    finally:
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    return results_by_index


rollout_requests = []
for prompt_group_index, prompt_row in enumerate(batch_rows, start=1):
    prompt_tokens = build_generation_prompt_tokens(tokenizer, prompt_row)
    rollout_requests.append(
        (
            prompt_tokens,
            group_size,
            types.SamplingParams(
                max_tokens=rl_max_tokens,
                temperature=rl_temperature,
                seed=seed + step * 100 + prompt_group_index,
                stop_token_ids=[tokenizer.eos_token_id],
            ),
        )
    )

rollout_results = asyncio.run(
    sample_many_async(
        sampler=sampler,
        sampling_requests=rollout_requests,
        max_in_flight=groups_per_batch,
    )
)
```

What this example guarantees:

If the script is async-heavy end to end, keep the outer CLI-friendly wrapper as:

```python
def main() -> int:
    return asyncio.run(main_async())
```

while putting the actual MinT workflow inside `async def main_async()`.

- This is the repo-standard train-side helper shape for bounded async sampling; copy it directly unless the experiment needs the fuller queue/worker engine.
- `groups_per_batch` is the closest Tinker / verl-style train knob here: it counts prompt groups per step; in simple helpers it bounds train-side prompt-group fan-out, and in the fuller queue/worker engine it can also match the rollout-worker count.
- The helper keeps at most `max_in_flight` live sampling tasks and immediately backfills a new request when one finishes.
- `results[index] = ...` preserves prompt order even when requests finish out of order.
- For eval, prefer the lighter semaphore-backed gather helper instead of reusing the train helper wholesale.

## Key rules for agents

- Always call `.result()` on sync Futures; for async APIs, follow the installed SDK's method-specific pattern. The docs consistently show single-await for client creation helpers, while request methods may be single-await or submit-then-await depending on SDK version
- Treat `save_weights_and_get_sampling_client(...)` as a live sampler-client helper, not a checkpoint-path API; if you need a durable sampler path, use `save_weights_for_sampler(...).result().path`
- For single-algorithm baselines, keep code in one Python file when possible: copy only a few core helpers from `tests/common.py`, do not import that file directly, and inline one-off RPC calls instead of wrapping them in tiny helper functions
- If you keep a timed async step context helper, prefer the readable name `smoke_step_async`
- Use a bounded sliding window for throughput-oriented train sampling, or `Semaphore(N)` + `gather` for lightweight eval — never unbounded `gather`
- Use client holder dict pattern when session recovery may recreate clients
- Overlap `forward_backward` and `optim_step` submissions when the installed SDK shape supports it and the added complexity is worth it
- Token alignment: next-token prediction shift `input=all[:-1], target=all[1:]`
- Prefer local Hugging Face tokenizer snapshots over `training_client.get_tokenizer()` in smoke tests and starter scripts; load from the cached snapshot path with `local_files_only=True` to avoid surprise Hub probes
- In this repo, starter scripts should read `MINT_BASE_URL` directly, keep experiment-local `.env` files minimal (`MINT_BASE_URL` / `MINT_API_KEY`), and default the base model in script args or code rather than through a shared `.env` contract
- When migrating a legacy script that already says `import tinker`, prefer the narrow bridge `import mint as tinker` first, then normalize endpoint and auth wiring onto `MINT_BASE_URL` and `MINT_API_KEY`; do not use that alias as the default in new scaffold-derived code
- LoRA LR is 20-100x higher than full fine-tuning LR
