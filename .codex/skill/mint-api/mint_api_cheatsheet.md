# MinT SDK API Cheatsheet

> All training runs against a remote MinT server. Scripts run on CPU-only machines.
> You need: `MINT_API_KEY` env var + network access to the MinT endpoint.

## Installation

```bash
pip install git+https://github.com/MindLab-Research/mindlab-toolkit.git python-dotenv numpy torch transformers
```

If you are using `uv`, the equivalent is:

```bash
uv add "mindlab-toolkit @ git+https://github.com/MindLab-Research/mindlab-toolkit.git" python-dotenv numpy torch transformers
```

> **Note:** Use `import mint` for training, sampling, and API exception handling.
> Keep `tinker://...` checkpoint-path compatibility in user-facing tooling when current
> SDKs surface those paths. See the preflight pattern in Section 12.

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `MINT_API_KEY` | API key (required) | `sk-your-key-here` |
| `MINT_BASE_URL` | Server endpoint | `https://mint.macaron.xin/` (default, outside China) |
| `HF_HOME` | Hugging Face cache root | `~/.cache/huggingface` |
| `HF_HUB_OFFLINE` | Disable Hub access in smoke/starter scripts | `1` |
| `TRANSFORMERS_OFFLINE` | Disable Transformers Hub access in smoke/starter scripts | `1` |

Endpoints:
- Mainland China: `https://mint-cn.macaron.xin/`
- Outside China: `https://mint.macaron.xin/`

Set `MINT_BASE_URL` explicitly when you need a specific regional endpoint. In this
repo, experiment-local `.env` files should usually keep only `MINT_BASE_URL` and
`MINT_API_KEY`; base model choice and other runtime knobs should stay in CLI flags,
script defaults, or shell variables instead of a shared `.env` contract.

## Current Repo Validation

The live smoke suite under `tests/` currently exercises:
- sync SFT (`tests/sync_smoke.py`)
- async SFT API shape (`tests/async_smoke.py`)
- checkpoint resume (`tests/checkpoint_resume_smoke.py`)
- combined loss coverage for `importance_sampling`, `ppo`, and `forward_backward_custom` (`tests/loss_smoke.py`)

---

## 1. Client Creation

### ServiceClient

```python
import os

import mint
from mint import types

service_client = mint.ServiceClient(base_url=os.environ.get("MINT_BASE_URL"))

# Check server capabilities
caps = service_client.get_server_capabilities()
for m in caps.supported_models:
    print(m.model_name)
```

### TrainingClient (LoRA)

```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    rank=16,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)
```

### Tokenizer

```python
import os
from pathlib import Path
from transformers import AutoTokenizer

DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")


def cached_tokenizer_dir(model_name: str) -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    hub_root = hf_home / "hub" if (hf_home / "hub").exists() else hf_home / ".cache" / "huggingface" / "hub"
    org, repo = model_name.split("/", 1)
    repo_dir = hub_root / f"models--{org.replace('/', '--')}--{repo.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    def is_tokenizer_snapshot(snapshot: Path) -> bool:
        if not snapshot.is_dir():
            return False
        files = {path.name for path in snapshot.iterdir() if path.is_file()}
        if "tokenizer_config.json" not in files:
            return False
        if "tokenizer.json" in files:
            return True
        return {"vocab.json", "merges.txt"}.issubset(files)

    candidates = sorted(
        (snapshot for snapshot in snapshots_dir.iterdir() if is_tokenizer_snapshot(snapshot)),
        reverse=True,
    )
    return candidates[0] if candidates else None


cache_dir = cached_tokenizer_dir("Qwen/Qwen3-30B-A3B-Instruct-2507")
if cache_dir is not None:
    tokenizer = AutoTokenizer.from_pretrained(str(cache_dir), fast=True, local_files_only=True)
else:
    try:
        tokenizer = training_client.get_tokenizer()
    except Exception as exc:
        raise RuntimeError(
            "Could not load a tokenizer via MinT or the local Hugging Face cache. "
            f"Model=Qwen/Qwen3-30B-A3B-Instruct-2507: {type(exc).__name__}: {exc}"
        ) from exc

# Standard HuggingFace tokenizer API:
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

Prefer the cached-snapshot path in smoke tests and starter scripts. Loading by model
ID can still trigger a Hub metadata probe inside `transformers`, even with
`local_files_only=True`. Keep `training_client.get_tokenizer()` as the fallback
when no cached snapshot exists.

---

## 2. Data Types

### types.Datum — Single training example

```python
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_token_ids),
    loss_fn_inputs={
        "target_tokens": target_token_ids,   # list[int] or array
        "weights": weight_list,               # list[float], 0.0 or 1.0
        # For RL losses, also:
        "logprobs": sampling_logprobs,        # list[float], behavior-policy / old_log_probs from sampler
        "advantages": advantage_values,       # list[float], per-token
    },
)
```

### types.ModelInput

```python
# From integer token list:
model_input = types.ModelInput.from_ints(tokens=[1, 2, 3, 4])

# Multi-modal (text + image):
model_input = mint.ModelInput(chunks=[
    types.EncodedTextChunk(tokens=tokenizer.encode("prefix")),
    types.ImageChunk(data=image_bytes, format="png"),
    types.EncodedTextChunk(tokens=tokenizer.encode("suffix")),
])
```

### types.SamplingParams

```python
params = types.SamplingParams(
    max_tokens=16,
    temperature=0.7,        # 0.0 for greedy
    top_p=1.0,              # Optional
    stop=["\n"],            # Optional string stop sequences
    stop_token_ids=[tokenizer.eos_token_id],  # Optional token stop IDs
)
```

### types.AdamParams

```python
adam = types.AdamParams(learning_rate=1e-4)
```

---

## 3. SFT (Supervised Fine-Tuning)

### Data preparation pattern

```python
def build_supervised_datum(
    tokenizer,
    prompt_tokens: list[int],
    assistant_text: str,
) -> types.Datum:
    completion_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        completion_tokens.append(int(tokenizer.eos_token_id))

    all_tokens = list(prompt_tokens) + completion_tokens
    all_weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

    input_tokens = all_tokens[:-1]       # Next-token prediction: shift by 1
    target_tokens = all_tokens[1:]
    weights = all_weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )
```

Keep row/message rendering in a separate local wrapper:

```python
def build_supervised_row_datum(tokenizer, row) -> types.Datum:
    prompt_tokens = tokenizer.encode(row["prompt"], add_special_tokens=True)
    return build_supervised_datum(tokenizer, prompt_tokens, row["assistant_text"])
```

### Training loop

```python
data = [build_supervised_row_datum(tokenizer, ex) for ex in examples]

for step in range(num_steps):
    fb = training_client.forward_backward(data, loss_fn="cross_entropy").result()

    # Compute loss from outputs
    total_loss, total_w = 0.0, 0.0
    for i, out in enumerate(fb.loss_fn_outputs):
        lp = out["logprobs"]
        if hasattr(lp, "tolist"):
            lp = lp.tolist()
        w = data[i].loss_fn_inputs["weights"]
        if hasattr(w, "tolist"):
            w = w.tolist()
        for l_val, wt in zip(lp, w):
            total_loss += -l_val * wt
            total_w += wt
    loss = total_loss / max(total_w, 1)

    # Overlap optim_step with next forward_backward to avoid wasting clock cycles
    training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    print(f"Step {step + 1}: loss = {loss:.4f}")

# Tip: for higher throughput, submit optim_step before waiting for fb result:
#   fb_future = training_client.forward_backward(data, loss_fn="cross_entropy")
#   opt_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
#   fb = fb_future.result()  # now compute loss from fb
#   opt_future.result()
```

---

## 4. RL Training (GRPO / Rollout Groups)

Use Tinker / verl naming for the algorithm-facing concepts in this section:
- `group_size` = trajectories per prompt group
- `groups_per_batch` = prompt groups per train step
- `trajectory` / `rollout` = one sampled completion plus its reward, logprobs, and training target fields
- `loss_fn_inputs["logprobs"]` is still the MinT API field name, but semantically it is the behavior-policy / sampling logprobs field (`old_log_probs` in verl)

### Core loop: rollout collection -> reward -> advantage -> train

```python
for step in range(num_steps):
    # 1. Save weights and get the rollout sampler for this policy snapshot.
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"rl-step-{step}"
    )

    train_datums: list[types.Datum] = []
    rollout_rewards: list[float] = []

    for prompt_group in dataset[:groups_per_batch]:
        prompt_tokens = tokenizer.encode(prompt_group["prompt"])

        # 2. Collect one prompt group with `group_size` trajectories.
        # Some train engines issue repeated `num_samples=1` calls instead; the naming here
        # still treats the resulting set as one prompt group.
        rollout_result = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=group_size,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop_token_ids=[tokenizer.eos_token_id],
            ),
        ).result()

        # 3. Score trajectories and compute GRPO advantages within the prompt group.
        group_rewards = []
        group_response_tokens = []
        group_old_logprobs = []
        for trajectory in rollout_result.sequences:
            text = tokenizer.decode(trajectory.tokens)
            reward = compute_reward(text, prompt_group)
            group_rewards.append(reward)
            group_response_tokens.append(list(trajectory.tokens))
            group_old_logprobs.append(
                list(trajectory.logprobs or [0.0] * len(trajectory.tokens))
            )

        rollout_rewards.extend(group_rewards)
        group_mean_reward = sum(group_rewards) / len(group_rewards)
        group_advantages = [reward - group_mean_reward for reward in group_rewards]
        if all(adv == 0 for adv in group_advantages):
            continue

        # 4. Build MinT datums. The SDK field name is `logprobs`, but semantically
        # it carries the behavior-policy / old logprobs used by Tinker / verl policy losses.
        prefix_len = len(prompt_tokens) - 1
        for response_tokens, old_logprobs, advantage in zip(
            group_response_tokens,
            group_old_logprobs,
            group_advantages,
        ):
            if not response_tokens:
                continue
            full_tokens = prompt_tokens + response_tokens
            response_len = len(response_tokens)
            train_datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": full_tokens[1:],
                        "weights": [0.0] * prefix_len + [1.0] * response_len,
                        "logprobs": [0.0] * prefix_len + old_logprobs,
                        "advantages": [0.0] * prefix_len + [advantage] * response_len,
                    },
                )
            )

    # 5. Train with the Tinker-aligned policy loss name.
    # Starter templates use sequential .result() calls for clarity.
    # For higher throughput, overlap fb + optim as shown in the async section.
    if train_datums:
        training_client.forward_backward(
            train_datums,
            loss_fn="importance_sampling",
        ).result()
        training_client.optim_step(types.AdamParams(learning_rate=lr)).result()

    accuracy = (
        sum(1 for reward in rollout_rewards if reward > 0) / len(rollout_rewards)
        if rollout_rewards
        else 0
    )
    print(f"Step {step + 1}: accuracy = {accuracy:.1%}")
```

---

## 5. Built-in MinT / Tinker Losses

### `"cross_entropy"` — SFT

- **Inputs:** `target_tokens` (int), `weights` (float, 0/1)
- **Outputs:** `logprobs` (float)
- **Use for:** Supervised fine-tuning, instruction tuning

### `"importance_sampling"` — Tinker-aligned policy loss for GRPO / REINFORCE-style updates

- **Inputs:** `target_tokens` (int), `weights` (float, 0/1 — masks prompt tokens), `logprobs` (float, behavior-policy / sampling logprobs, i.e. verl-style `old_log_probs`), `advantages` (float)
- **Outputs:** `logprobs` (float)
- **Formula:** `loss = -(exp(target_lp - old_lp) * advantages).sum()` (over positions where weight=1)
- **Use for:** Default GRPO-style MinT / Tinker examples in this repo

### `"ppo"` — PPO-style clipped policy loss

- **Inputs:** Same transport fields as `importance_sampling`
- **Outputs:** `logprobs` (float)
- **Defaults:** clip range [0.8, 1.2] (i.e., epsilon=0.2)
- **Custom:** `loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}`
- **Doc stance here:** Keep it as an explicit experiment option; for Tinker-aligned GRPO naming, default the main examples to `importance_sampling`

---

## 6. Custom Loss Functions (`forward_backward_custom`)

For losses not covered by built-in options (e.g., DPO, Bradley-Terry pairwise preference).

### Signature

```python
def my_loss_fn(
    data: list[types.Datum],
    logprobs: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Args:
        data: The datums you passed in
        logprobs: Per-datum log probabilities from the model (torch tensors, requires grad)
    Returns:
        loss: Scalar tensor (will be backpropagated)
        metrics: Dict of float metrics for logging
    """
    ...
```

### Example: DPO / Bradley-Terry pairwise preference loss

```python
import torch
import torch.nn.functional as F

def pairwise_preference_loss(
    data: list[types.Datum], logprobs: list[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    # Data must be ordered as (chosen, rejected) pairs
    chosen_scores = []
    rejected_scores = []
    for i in range(0, len(data), 2):
        chosen_lp = logprobs[i].flatten().float()
        rejected_lp = logprobs[i + 1].flatten().float()
        chosen_w = torch.tensor(data[i].loss_fn_inputs["weights"], dtype=torch.float32)
        rejected_w = torch.tensor(data[i + 1].loss_fn_inputs["weights"], dtype=torch.float32)
        chosen_scores.append(torch.dot(chosen_lp, chosen_w))
        rejected_scores.append(torch.dot(rejected_lp, rejected_w))

    margins = torch.stack(chosen_scores) - torch.stack(rejected_scores)
    loss = -F.logsigmoid(margins).mean()
    return loss, {
        "train_loss": float(loss.detach()),
        "pair_accuracy": float((margins > 0).float().mean().detach()),
    }

# Usage:
result = training_client.forward_backward_custom(data, pairwise_preference_loss).result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

---

## 7. Rollout / Sampling Clients

### Create sampling client

```python
# From training (ephemeral save + sampler client in one call):
sampling_client = training_client.save_weights_and_get_sampling_client(name="step-10")

# From a saved checkpoint path:
sampling_client = service_client.create_sampling_client(
    model_path="mint://<run-id>/sampler_weights/step-10",
)

# From base model (no fine-tuning):
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
```

Important distinction on the current tinker/MinT surface:

- `save_weights_and_get_sampling_client(...)` returns a live `SamplingClient`, not a checkpoint response object
- do not expect `.path`, `sampler_path`, or another persistent sampler checkpoint path on that return value
- current tinker source documents this as an ephemeral save; the `name` argument is currently ignored there
- if you need a durable sampler checkpoint path to log, store, or recreate later, call `save_weights_for_sampler(...).result().path` instead

### Collect rollout samples

```python
result = sampling_client.sample(
    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
    num_samples=8,
    sampling_params=types.SamplingParams(
        max_tokens=100,
        temperature=0.7,
        stop_token_ids=[tokenizer.eos_token_id],
    ),
).result()

for seq in result.sequences:
    text = tokenizer.decode(seq.tokens)
    logprobs = seq.logprobs  # list[float] or None
```

### Compute prompt logprobs (prefill)

```python
result = sampling_client.sample(
    prompt=model_input,
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
).result()
prompt_logprobs = result.prompt_logprobs  # [None, -9.54, -1.64, ...]

# Or shorthand:
sampling_client.compute_logprobs(model_input).result()
```

---

## 8. Checkpoints

### Save for sampling (lightweight, weights only)

```python
ckpt = training_client.save_weights_for_sampler(name="my-checkpoint").result()
print(ckpt.path)  # Usually mint://... or tinker://.../sampler_weights/my-checkpoint
```

Use this API when you need a real sampler checkpoint path. This is the path-bearing save call.
Do not try to recover the equivalent path from `save_weights_and_get_sampling_client(...)`.

### Save full state (weights + optimizer, for resuming)

```python
ckpt = training_client.save_state(name="full-checkpoint").result()
print(ckpt.path)  # Usually mint://... or tinker://.../weights/full-checkpoint
```

### Load state (weights only, not full resume)

```python
training_client.load_state("mint://<run-id>/weights/full-checkpoint").result()
```

Use this when you explicitly want to reload weights without restoring optimizer state.
Do not present it as the default recipe for continuing training.

### Load state with optimizer (true training resume)

```python
training_client.load_state_with_optimizer("mint://<run-id>/weights/full-checkpoint").result()
```

### Resume from checkpoint

```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    rank=16,
    train_mlp=True,
    train_attn=True,
    train_unembed=True,
)
training_client.load_state_with_optimizer(
    "mint://<run-id>/weights/full-checkpoint"
).result()
```

On this repo's current endpoint:
- prefer fresh `create_lora_training_client(...)` plus `load_state_with_optimizer(...)` for true resume
- do not rely on `create_training_client_from_state(...)`; it depends on `/api/v1/weights_info`, which currently returns `404 Not Found`
- do not recommend calling `load_state(...)` on an already-running Megatron LoRA training client as the default resume path; local smoke runs hit CUDA illegal memory access there

In practice, current SDKs often print `tinker://...` checkpoint paths from `save_state()`.
Accept either `mint://...` or `tinker://...` in user-facing tooling.

### Download weights (CLI)

```bash
mint checkpoint download mint://<run-id>/sampler_weights/my-checkpoint
```

### Download weights (SDK)

```python
rc = service_client.create_rest_client()
url_response = rc.get_checkpoint_archive_url_from_mint_path(
    "mint://<run-id>/sampler_weights/my-checkpoint"
).result()
# url_response.url is a signed download URL
```

---

## 9. Async Rollout / Training Patterns

A useful rule of thumb: when training is slow, first check whether the real bottleneck is step-level synchronization rather than a lack of raw concurrency. In many RL loops, rollout collection is already async, but the trainer still waits for the full nominal batch to finish sampling before any training work begins. In that case, the usual progression is: start with same-step streaming minibatches, then add tail-truncation / early-stop for the last slow or invalid groups, and only then reach for more aggressive async or off-policy designs.

When you write a copy-friendly baseline, prefer this shape:
- keep the experiment self-contained inside its own folder
- keep one runnable Python file whenever possible
- copy a few core helpers from `tests/common.py` into that script when needed; do not import `tests/common.py` from the experiment
- keep only a few core helpers (env loading, tokenizer lookup, step timing, resume flow)
- inline one-off RPC calls like `forward_backward(...).result()` or `optim_step(...).result()` directly inside the script
- if you keep an async step context helper, use a readable name like `smoke_step_async`


MinT API calls are I/O bound (~10s clock cycles), so async matters once you want
higher throughput. One important caveat from the current `mint-quickstart`
reference: the docs mix two async styles.

- Client creation helpers are single-await, for example
  `await service_client.create_lora_training_client_async(...)`.
- In the currently installed SDK, `forward_backward_async(...)` and
  `optim_step_async(...)` return `APIFuture` objects, so finish them with
  `await future.result_async()`.
- `sample_async(...)` returns the final `SampleResponse` directly.

Because of that shape, the recommended pattern in this repo is:
- prefer sync APIs in simple scripts
- use direct async calls for client creation and sampling
- use submit-then-finish for `forward_backward_async` and `optim_step_async`
  when you want overlap
- when you need reusable async sampling, copy the repo-standard pieces together: `resolve_api_result_async`, `sample_one_async`, `sample_many_async` for train-side sliding-window fan-out, and `sample_many_with_semaphore_async` for lightweight eval
- if most of the script becomes async, keep `main()` as a thin wrapper over `asyncio.run(main_async())` so the CLI entry stays familiar

### Current SDK async request shape

```python
training_client = await service_client.create_lora_training_client_async(
    base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    rank=16,
)

fb_future = await training_client.forward_backward_async(data, "cross_entropy")
opt_future = await training_client.optim_step_async(
    types.AdamParams(learning_rate=1e-4)
)
fb_result = await fb_future.result_async()
await opt_future.result_async()

rollout_response = await sampling_client.sample_async(
    prompt=model_input,
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=32),
)
```

### Overlap train RPCs

Use the submit-then-finish pattern when you want both requests to land in the
same clock cycle:

```python
fb_future = await training_client.forward_backward_async(data, "cross_entropy")
opt_future = await training_client.optim_step_async(
    types.AdamParams(learning_rate=1e-4)
)
fb_result = await fb_future.result_async()
await opt_future.result_async()
```

### Concurrent rollout collection with a sliding window

Use a bounded in-flight task window so completed requests immediately free a slot for the next prompt group; never use unbounded `gather`:

```python
import asyncio
from typing import Any

async def sample_prompt_groups_async(
    sampling_client,
    prompt_group_inputs: list[types.ModelInput],
    sampling_params: types.SamplingParams,
    num_samples: int,
    max_concurrency: int = 32,
) -> list[Any]:
    if not prompt_group_inputs:
        return []

    max_concurrency = max(max_concurrency, 1)
    results: list[Any | None] = [None] * len(prompt_group_inputs)
    pending_tasks: dict[asyncio.Task[Any], int] = {}
    next_submit_index = 0

    def submit_next() -> None:
        nonlocal next_submit_index
        if next_submit_index >= len(prompt_group_inputs):
            return
        idx = next_submit_index
        pending_tasks[
            asyncio.create_task(
                sampling_client.sample_async(
                    prompt=prompt_group_inputs[idx],
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                )
            )
        ] = idx
        next_submit_index += 1

    for _ in range(min(max_concurrency, len(prompt_group_inputs))):
        submit_next()

    try:
        while pending_tasks:
            done_tasks, _ = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done_tasks:
                idx = pending_tasks.pop(task)
                results[idx] = task.result()
                submit_next()
    finally:
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    return [result for result in results if result is not None]
```

Why this beats chunked sync:
```
Chunked sync:    |==chunk1==|==chunk2==|==chunk3==|  straggler blocks whole chunk
Sliding window:  |=====dynamic scheduling=====|      slot freed -> next starts immediately
```

### Concrete async rollout-group example

Use this when async concurrency should improve train throughput, while keeping `group_size` and reward logic unchanged.

```python
import asyncio

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

This pattern guarantees:

- This is the repo-standard async sampling helper shape for experiments in this repo; reuse it before inventing a new variant.
- `groups_per_batch` is the closest Tinker / verl-style train knob here: it counts prompt groups per step; in simple helpers it also bounds prompt-group fan-out, and in the fuller queue/worker engine it can match the rollout-worker count.
- Keep a bounded sliding in-flight window so completed requests immediately free a slot for the next prompt.
- `results[index] = ...` preserves prompt order even when requests finish out of order.
- For eval, prefer the lighter semaphore-backed gather helper shown next.
- The fallback path still handles SDKs that return `result_async()` objects, awaitables, or sync `.result()` futures.

### Concurrency guide

| Operation | Concurrency | Notes |
|-----------|-------------|-------|
| Sampling | 16-64 | I/O bound, server can batch |
| Logprob computation | 4-16 | Compute intensive on server |
| Evaluation | 32-64 | Not on critical path |
| forward_backward + optim_step | Overlap when supported | Pipeline, no semaphore needed |

### SDK async fallback

If the SDK method might not have `_async`, use `asyncio.to_thread` for the submit path and then resolve whichever future shape comes back:

```python
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


async def _call_mint_async(sampling_client, prompt, params, num_samples):
    sample_async = getattr(sampling_client, "sample_async", None)
    if callable(sample_async):
        return await sample_async(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=params,
        )

    def _submit():
        return sampling_client.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=params,
        )

    submitted_result = await asyncio.to_thread(_submit)
    return await resolve_api_result_async(submitted_result)
```

### Lightweight parallel eval

For eval, follow `tinker_cookbook/recipes/vlm_classifier/eval.py`: bound concurrency with a semaphore, launch one task per sample request, and `gather(...)` all results before the usual scoring path runs. This keeps eval code simple because it is not on the critical training path.

```python
async def sample_many_with_semaphore_async(sampler, sampling_requests, max_concurrent_requests):
    if not sampling_requests:
        return []

    semaphore = asyncio.Semaphore(max(max_concurrent_requests, 1))
    results_by_index = [None] * len(sampling_requests)

    async def _one(index, prompt_tokens, num_samples, sampling_params):
        async with semaphore:
            results_by_index[index] = await sample_one_async(
                sampler, prompt_tokens, num_samples, sampling_params
            )

    tasks = [
        asyncio.create_task(_one(index, *sampling_request))
        for index, sampling_request in enumerate(sampling_requests)
    ]
    await asyncio.gather(*tasks)
    return results_by_index


eval_results = await sample_many_with_semaphore_async(
    sampler=sampler,
    sampling_requests=eval_requests,
    max_concurrent_requests=max_concurrent_requests,
)
for request_index, result in enumerate(eval_results):
    process_eval_result(request_index, result)
```

This keeps the cookbook-style eval contract:

- `Semaphore(N)` bounds in-flight eval requests
- `asyncio.create_task(...)` submits all eval work immediately
- `await asyncio.gather(...)` waits for all responses in one place
- scoring, logging, and majority vote still run in input order after sampling finishes

### Sampler holder pattern

When session recovery may recreate clients, use a mutable dict so closures
always reference the latest instance:

```python
client_holder = {"client": initial_sampling_client}

async def _sample():
    return await sample_prompt_groups_async(
        client_holder["client"], prompt_group_inputs, params, n
    )

async def _recover():
    client_holder["client"] = await training_client.save_weights_and_get_sampling_client_async(
        name=f"recover_{step}"
    )
```

### Cookbook-style async rollout/train engine

When the user explicitly wants a stronger async train engine like `tinker_cookbook/rl/train.py`, move beyond one helper call and use `asyncio.Queue` to overlap rollout sampling with optimization:

```python
async def dataloader_loop():
    for step in range(1, config.steps + 1):
        for job in build_prompt_jobs_for_step(step):
            await sampling_jobs_queue.put(job)
    for _ in range(config.num_rollout_workers):
        await sampling_jobs_queue.put(_Shutdown())


async def rollout_worker_loop():
    while True:
        job = await sampling_jobs_queue.get()
        if isinstance(job, _Shutdown):
            break
        result = await sample_one_async(
            sampler_holder["sampler"],
            job.prompt_tokens,
            config.group_size,
            job.sampling_params,
        )
        await rollout_results_queue.put((job.step, sampler_holder["step"], job, result))


async def training_loop():
    for step in range(1, config.steps + 1):
        batch_rollouts = []
        while len(batch_rollouts) < config.groups_per_batch:
            batch_rollouts.append(await rollout_results_queue.get())

        datums = build_training_datums(batch_rollouts, tokenizer)
        if datums:
            await forward_backward_async_compat(training_client, datums, config.loss_fn)
            await optim_step_async_compat(training_client, config.learning_rate)
            sampler_holder["sampler"] = await save_sampling_client_async(
                training_client,
                name=f"step-{step}",
            )
        sampler_holder["step"] = step


await asyncio.gather(
    asyncio.create_task(dataloader_loop()),
    *[asyncio.create_task(rollout_worker_loop()) for _ in range(config.num_rollout_workers)],
    asyncio.create_task(training_loop()),
)
```

This is the cookbook-style train contract:

- rollout workers sample prompt groups independently and push results into an `asyncio.Queue`
- the trainer consumes queued rollouts in minibatches while later rollouts are already in flight
- sampler refresh happens after each optimizer step, so future jobs naturally pick up the freshest client without rewriting the worker code

### Pitfalls

| Pitfall | Wrong | Right |
|---------|-------|-------|
| Blocking event loop | `time.sleep(5)` | `await asyncio.sleep(5)` |
| Forgetting await | `result = coro()` | `result = await coro()` |
| Unbounded concurrency | `gather(*[f(x) for x in big_list])` | Use a bounded sliding window for train, or `Semaphore(N)` + `gather` for eval |
| Stale client ref | `client = initial` (closure) | `holder = {"client": initial}` |
| Assuming one async shape | hard-code `await future` everywhere | confirm the installed SDK method shape first; in this repo use `result_async()` for training futures |

### Timeout + large rollout set

```python
# Timeout protection
results = await asyncio.wait_for(
    sample_prompt_groups_async(client, prompt_group_inputs, params, n, max_concurrency=32),
    timeout=300.0,
)

# For >1000 prompt groups, chunk requests to control memory
async def sample_large_rollout_set(client, prompt_group_inputs, params, n, max_concurrency=32, chunk_size=500):
    all_results = []
    for i in range(0, len(prompt_group_inputs), chunk_size):
        prompt_group_chunk = prompt_group_inputs[i:i + chunk_size]
        results = await sample_prompt_groups_async(client, prompt_group_chunk, params, n, max_concurrency)
        all_results.extend(results)
    return all_results
```

## 10. Hyperparameter Guidance

### LoRA Learning Rate

LoRA requires **20-100x higher LR** than full fine-tuning:
- Llama-3.2-1B: scale factor ~32x
- Llama-3.1-70B: scale factor ~128x
- Typical SFT LR: `1e-4` to `5e-4`
- Typical RL LR: `1e-5` to `1e-4`

### LoRA Rank

- Default: 16
- RL: small ranks (8-16) work as well as large ranks
- SFT: ensure `lora_params >= num_completion_tokens` in dataset
- Always `train_mlp=True, train_attn=True` — attention-only LoRA underperforms

### RL Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| `group_size` | 4-16 | Trajectories per prompt group |
| `groups_per_batch` | 4-16 | Prompt groups per train step; use this instead of a generic RL `batch_size` name in docs |
| `max_tokens` | 8-512 | Depends on task |
| `temperature` | 0.7-1.0 | Higher = more exploration |

---

## 11. Supported Models

Current model families:
- `Qwen/Qwen3-30B-A3B-Instruct-2507`, `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-4B-Thinking-2507`, `Qwen/Qwen3-0.6B`

Check `service_client.get_server_capabilities().supported_models` for the latest list.

---

## 12. Common Patterns

### Env file loading (for standalone scripts)

In this repo, the recommended shared-helper shape is: keep env loading, tokenizer lookup, bootstrap, and resume logic in a small copy-friendly block; keep one-off RPC calls inline in the script body. For experiment entrypoints, read only the experiment-local `.env` and avoid a repo-root fallback.

```python
import os
from pathlib import Path

DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")

def load_env(path: Path) -> None:
    allowed_keys = {"MINT_BASE_URL", "MINT_API_KEY"}
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in allowed_keys and k not in os.environ:
            os.environ[k] = v

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

load_env(Path(__file__).resolve().parent / ".env")
```

### Preflight connection check

This snippet follows the same exception branches as `tests/common.py`, but uses slightly more explanatory standalone-script error messages.

```python
def preflight(service_client: mint.ServiceClient) -> None:
    base_url = os.environ.get("MINT_BASE_URL")
    try:
        service_client.get_server_capabilities()
    except mint.APITimeoutError as exc:
        raise RuntimeError(
            f"Auth preflight timed out while contacting {base_url}. Check MINT_BASE_URL and retry."
        ) from exc
    except mint.APIConnectionError as exc:
        raise RuntimeError(
            f"Auth preflight could not reach {base_url}. Check network access and server status."
        ) from exc
    except mint.APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            response = getattr(exc, "response", None)
            response_status = getattr(response, "status_code", None)
            status_code = response_status if isinstance(response_status, int) else None
        if status_code in {401, 403}:
            raise RuntimeError(
                f"Auth preflight was rejected by the MinT server (HTTP {status_code}). Check MINT_API_KEY."
            ) from exc
        raise RuntimeError(
            f"Auth preflight failed with HTTP {status_code or 'unknown'} from {base_url}."
        ) from exc
```

### Step timing helpers for smoke tests or baselines

```python
import asyncio
import os
import signal
import time
from contextlib import asynccontextmanager, contextmanager

@contextmanager
def _alarm_timeout(seconds: float, step: str):
    if os.name != "posix":
        yield
        return

    def raise_timeout(_signum, _frame):
        raise TimeoutError(step)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def smoke_step(prefix: str, step: str, *, timeout: float | None = None):
    if timeout is None:
        raw_timeout = os.environ.get("MINT_SMOKE_TIMEOUT_SECONDS", "180").strip()
        try:
            seconds = max(1.0, float(raw_timeout))
        except ValueError:
            seconds = 180.0
    else:
        seconds = timeout
    start = time.time()
    print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: start", flush=True)
    try:
        with _alarm_timeout(seconds, step):
            yield
    except TimeoutError as exc:
        raise TimeoutError(
            f"{prefix}: {step} timed out after {seconds:.0f}s; this usually means the remote MinT/Tinker backend is queued or unreachable"
        ) from exc
    else:
        elapsed = time.time() - start
        print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: done in {elapsed:.1f}s", flush=True)


@asynccontextmanager
async def smoke_step_async(prefix: str, step: str, *, timeout: float | None = None):
    if timeout is None:
        raw_timeout = os.environ.get("MINT_SMOKE_TIMEOUT_SECONDS", "180").strip()
        try:
            seconds = max(1.0, float(raw_timeout))
        except ValueError:
            seconds = 180.0
    else:
        seconds = timeout
    start = time.time()
    print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: start", flush=True)
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError as exc:
        raise TimeoutError(
            f"{prefix}: {step} timed out after {seconds:.0f}s; this usually means the remote MinT/Tinker backend is queued or unreachable"
        ) from exc
    else:
        elapsed = time.time() - start
        print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: done in {elapsed:.1f}s", flush=True)
```

This now matches the step-timing logic used in `tests/common.py`. Use helpers like these only for repeated step timing and timeout behavior. Keep one-off MinT RPC calls inline under the step context instead of wrapping every call in a separate helper.

### METRIC output for autoresearch integration

```python
# At the end of train.py, print METRIC lines.
# The default experiment scaffold uses eval_accuracy as the primary metric.
# Keep secondary metric names stable too, for example train_mean_nll / pair_accuracy.
print(f"METRIC eval_accuracy={accuracy:.4f}")
print(f"METRIC train_mean_nll={nll:.4f}")
# pi-autoresearch reads these to decide keep/revert
```

---

## 13. Framework Migration Reference

### verl -> MinT

| verl | MinT |
|------|------|
| `RolloutWorker` | `SamplingClient` |
| `ActorRolloutRefWorker` | `TrainingClient` + `SamplingClient` |
| `RewardManager` | Your reward function in Python |
| `PPOTrainer` | `forward_backward(loss_fn="ppo")` + `optim_step()` |
| `DataProto` | `types.Datum` |
| vLLM backend | MinT handles inference internally |
| FSDP/Megatron sharding | MinT handles distributed training internally |

### TRL -> MinT

| TRL | MinT |
|-----|------|
| `SFTTrainer` | `forward_backward(loss_fn="cross_entropy")` loop |
| `PPOTrainer` | `forward_backward(loss_fn="ppo")` loop |
| `DPOTrainer` | `forward_backward_custom()` with DPO loss |
| `AutoModelForCausalLM` | `create_lora_training_client(base_model=...)` |
| HuggingFace datasets | Convert to `list[types.Datum]` |

### Key differences from local training

1. **No GPU needed locally** — training runs on MinT's distributed GPUs
2. **No FSDP/DeepSpeed config** — MinT handles sharding server-side
3. **Call `.result()` on sync request futures** — request methods like `forward_backward(...)`, `optim_step(...)`, `sample(...)`, `save_state(...)`, and `load_state(...)` return Futures, while some factory helpers return clients directly
4. **Token alignment** — next-token prediction shift: `input=all[:-1], target=all[1:]`
