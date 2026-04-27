#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

from common import (
    add_smoke_cli_args,
    bootstrap_training_test,
    resume_training_client_from_state,
    smoke_config_from_args,
    smoke_step,
)

TEST_NAME = "loss smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MinT policy/custom loss smoke test.")
    add_smoke_cli_args(parser)
    return parser.parse_args()


def _build_rl_datums(ctx, *, prompt: str) -> tuple[list[int], list]:
    from mint import types

    prompt_tokens = ctx.tokenizer.encode(prompt, add_special_tokens=True)

    with smoke_step(TEST_NAME, "save weights and create sampling client for RL losses"):
        sampler = ctx.training_client.save_weights_and_get_sampling_client(
            name=f"loss-sampler-{time.time_ns()}"
        )

    with smoke_step(TEST_NAME, "sample RL group for policy losses"):
        result = sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=2,
            sampling_params=types.SamplingParams(
                max_tokens=16,
                temperature=0.7,
                stop_token_ids=[ctx.tokenizer.eos_token_id],
            ),
        ).result()
    sequences = list(getattr(result, "sequences", []) or [])
    if len(sequences) < 2:
        raise RuntimeError("policy loss smoke expected at least 2 sampled sequences")

    datums: list[types.Datum] = []
    prefix_len = len(prompt_tokens) - 1
    centered_advantages = [0.5, -0.5]
    for index, (sequence, advantage) in enumerate(zip(sequences[:2], centered_advantages), start=1):
        response_tokens = list(sequence.tokens)
        if not response_tokens:
            raise RuntimeError(f"Sample {index} returned no tokens")
        response_text = ctx.tokenizer.decode(response_tokens).strip()
        print(f"{TEST_NAME}: policy_sample_{index}={response_text!r}")
        logprobs = list(sequence.logprobs or [0.0] * len(response_tokens))
        full_tokens = prompt_tokens + response_tokens
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": full_tokens[1:],
                    "weights": [0.0] * prefix_len + [1.0] * len(response_tokens),
                    "logprobs": [0.0] * prefix_len + logprobs,
                    "advantages": [0.0] * prefix_len + [advantage] * len(response_tokens),
                },
            )
        )

    return prompt_tokens, datums


def _run_policy_loss_smoke(
    ctx,
    *,
    loss_fn: str,
    loss_fn_config: dict[str, float] | None = None,
    resume_after: bool = False,
) -> None:
    from mint import types

    _, datums = _build_rl_datums(ctx, prompt="Question: What is 4 + 6?\nAnswer:")

    with smoke_step(TEST_NAME, f"run forward_backward {loss_fn}"):
        fb = ctx.training_client.forward_backward(
            datums,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        ).result()
    if not getattr(fb, "loss_fn_outputs", None):
        raise RuntimeError(f"{loss_fn} forward_backward returned no loss outputs")

    with smoke_step(TEST_NAME, f"run optim_step after {loss_fn}"):
        ctx.training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

    if not resume_after:
        return

    with smoke_step(TEST_NAME, f"save training state after {loss_fn}"):
        state_path = ctx.training_client.save_state(name=f"{loss_fn}-state-{time.time_ns()}").result().path
    print(f"{TEST_NAME}: {loss_fn}_state_path={state_path}")

    resumed_client, resume_mode = resume_training_client_from_state(
        TEST_NAME,
        ctx.service_client,
        ctx.model,
        state_path,
        config=ctx.config,
    )
    print(f"{TEST_NAME}: {loss_fn}_resume_mode={resume_mode}")

    with smoke_step(TEST_NAME, f"run resumed forward_backward {loss_fn}"):
        resumed_fb = resumed_client.forward_backward(
            datums,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        ).result()
    if not getattr(resumed_fb, "loss_fn_outputs", None):
        raise RuntimeError(f"resumed {loss_fn} forward_backward returned no loss outputs")

    with smoke_step(TEST_NAME, f"run resumed optim_step after {loss_fn}"):
        resumed_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()


def run_custom_loss_smoke(ctx) -> None:
    import torch
    import torch.nn.functional as F
    from mint import types

    prompt = "Explain why regular backups matter."
    prompt_tokens = ctx.tokenizer.encode(f"User: {prompt}\nAssistant:", add_special_tokens=True)

    def build_datum(completion_text: str) -> types.Datum:
        completion_tokens = ctx.tokenizer.encode(f" {completion_text}", add_special_tokens=False)
        completion_tokens.append(ctx.tokenizer.eos_token_id)
        all_tokens = prompt_tokens + completion_tokens
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": all_tokens[1:],
                "weights": [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens),
            },
        )

    data = [
        build_datum("Regular backups reduce recovery time after mistakes or outages."),
        build_datum("Backups are good."),
    ]

    def to_tensor(value):
        if hasattr(value, "to_torch"):
            return value.to_torch().flatten().float()
        if hasattr(value, "tolist"):
            return torch.tensor(value.tolist(), dtype=torch.float32).flatten()
        return torch.tensor(value, dtype=torch.float32).flatten()

    def pairwise_preference_loss(batch, logprobs_list):
        chosen = torch.dot(logprobs_list[0].flatten().float(), to_tensor(batch[0].loss_fn_inputs["weights"]))
        rejected = torch.dot(logprobs_list[1].flatten().float(), to_tensor(batch[1].loss_fn_inputs["weights"]))
        margin = chosen - rejected
        loss = -F.logsigmoid(margin)
        return loss, {
            "train_loss": float(loss.detach().cpu()),
            "pair_accuracy": float((margin > 0).float().detach().cpu()),
        }

    with smoke_step(TEST_NAME, "run forward_backward_custom"):
        result = ctx.training_client.forward_backward_custom(data, pairwise_preference_loss).result()
    metrics = getattr(result, "metrics", {}) or {}
    with smoke_step(TEST_NAME, "run optim_step after custom loss"):
        ctx.training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

    print(f"{TEST_NAME}: custom_loss_metrics={metrics}")
    if "pair_accuracy" not in metrics:
        raise RuntimeError("forward_backward_custom did not return pair_accuracy metric")


def main() -> int:
    args = parse_args()
    ctx = bootstrap_training_test(TEST_NAME, config=smoke_config_from_args(args))
    _run_policy_loss_smoke(ctx, loss_fn="importance_sampling", resume_after=True)
    _run_policy_loss_smoke(ctx, loss_fn="ppo", resume_after=True)
    run_custom_loss_smoke(ctx)
    print(f"{TEST_NAME}: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{TEST_NAME}: FAIL: {exc}", file=sys.stderr)
        raise
