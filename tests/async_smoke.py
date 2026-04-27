#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from common import (
    add_smoke_cli_args,
    bootstrap_training_test_async,
    build_sft_datum,
    smoke_config_from_args,
    smoke_step_async,
)

TEST_NAME = "async smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the asynchronous MinT smoke test.")
    add_smoke_cli_args(parser)
    return parser.parse_args()


async def resolve_api_result_async(maybe_result):
    """Normalize async SDK wrappers into the final payload."""
    if hasattr(maybe_result, "result_async") and callable(maybe_result.result_async):
        return await resolve_api_result_async(await maybe_result.result_async())
    if asyncio.isfuture(maybe_result):
        return await resolve_api_result_async(await maybe_result)
    if hasattr(maybe_result, "__await__"):
        return await resolve_api_result_async(await maybe_result)
    if hasattr(maybe_result, "result") and callable(maybe_result.result):
        return await resolve_api_result_async(await asyncio.to_thread(maybe_result.result))
    return maybe_result


async def main_async() -> int:
    from mint import types

    args = parse_args()
    ctx = await bootstrap_training_test_async(TEST_NAME, config=smoke_config_from_args(args))
    datum = build_sft_datum(
        prompt="Question: What is 5 + 6?\nAnswer:",
        completion=" 11",
        tokenizer=ctx.tokenizer,
        types_module=types,
    )

    async with smoke_step_async(TEST_NAME, "get training info_async"):
        info = await ctx.training_client.get_info_async()
    model_name = getattr(getattr(info, "model_data", None), "model_name", None)
    if not model_name:
        raise RuntimeError("training_client.get_info_async() did not return model_data.model_name")
    print(f"{TEST_NAME}: info_model_name={model_name}")

    async with smoke_step_async(TEST_NAME, "submit forward_async"):
        forward_future = await ctx.training_client.forward_async([datum], "cross_entropy")
    async with smoke_step_async(TEST_NAME, "finish forward_async"):
        forward_result = await forward_future.result_async()
    if not getattr(forward_result, "loss_fn_outputs", None):
        raise RuntimeError("forward_async returned no loss outputs")

    async with smoke_step_async(TEST_NAME, "submit forward_backward_async"):
        fb_future = await ctx.training_client.forward_backward_async([datum], "cross_entropy")
    async with smoke_step_async(TEST_NAME, "submit optim_step_async"):
        opt_future = await ctx.training_client.optim_step_async(types.AdamParams(learning_rate=5e-5))
    async with smoke_step_async(TEST_NAME, "finish forward_backward_async"):
        fb_result = await fb_future.result_async()
    async with smoke_step_async(TEST_NAME, "finish optim_step_async"):
        await opt_future.result_async()
    if not getattr(fb_result, "loss_fn_outputs", None):
        raise RuntimeError("forward_backward_async returned no loss outputs")

    async with smoke_step_async(TEST_NAME, "save weights and create sampling client"):
        sampler = await ctx.training_client.save_weights_and_get_sampling_client_async(
            name=f"async-sampler-{time.time_ns()}"
        )
    async with smoke_step_async(TEST_NAME, "get base model_async from in-process sampler"):
        sampler_base_model = await sampler.get_base_model_async()
    if not sampler_base_model:
        raise RuntimeError("sampler.get_base_model_async() returned an empty model name")
    print(f"{TEST_NAME}: sampler_base_model={sampler_base_model}")
    prompt_tokens = ctx.tokenizer.encode("Question: What is 8 + 9?\nAnswer:", add_special_tokens=True)
    async with smoke_step_async(TEST_NAME, "sample_async"):
        sample = await sampler.sample_async(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=8,
                temperature=0.0,
                stop_token_ids=[ctx.tokenizer.eos_token_id],
            ),
        )
    if not getattr(sample, "sequences", None):
        raise RuntimeError("sample_async returned no sequences")
    sampled_text = ctx.tokenizer.decode(sample.sequences[0].tokens).strip()
    print(f"{TEST_NAME}: sampled_text={sampled_text!r}")

    async with smoke_step_async(TEST_NAME, "compute_logprobs_async"):
        prompt_logprobs = await sampler.compute_logprobs_async(types.ModelInput.from_ints(tokens=prompt_tokens))
    if prompt_logprobs is None:
        raise RuntimeError("compute_logprobs_async returned None")

    async with smoke_step_async(TEST_NAME, "save weights for external sampler_async"):
        sampler_payload = await resolve_api_result_async(
            ctx.training_client.save_weights_for_sampler_async(
                name=f"async-external-sampler-{time.time_ns()}"
            )
        )
    sampler_path = getattr(sampler_payload, "path", None)
    if not sampler_path:
        raise RuntimeError("save_weights_for_sampler_async returned no sampler path")
    print(f"{TEST_NAME}: sampler_path={sampler_path}")
    async with smoke_step_async(TEST_NAME, "create external sampling client_async"):
        external_sampler = await resolve_api_result_async(
            ctx.service_client.create_sampling_client_async(model_path=sampler_path)
        )
    async with smoke_step_async(TEST_NAME, "get base model_async from external sampler"):
        external_base_model = await external_sampler.get_base_model_async()
    if not external_base_model:
        raise RuntimeError("external_sampler.get_base_model_async() returned an empty model name")
    print(f"{TEST_NAME}: external_base_model={external_base_model}")
    async with smoke_step_async(TEST_NAME, "sample_async from external sampler"):
        external_sample = await external_sampler.sample_async(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=8,
                temperature=0.0,
                stop_token_ids=[ctx.tokenizer.eos_token_id],
            ),
        )
    if not getattr(external_sample, "sequences", None):
        raise RuntimeError("external sample_async returned no sequences")
    external_sampled_text = ctx.tokenizer.decode(external_sample.sequences[0].tokens).strip()
    print(f"{TEST_NAME}: external_sampled_text={external_sampled_text!r}")
    print(f"{TEST_NAME}: PASS")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{TEST_NAME}: FAIL: {exc}", file=sys.stderr)
        raise
