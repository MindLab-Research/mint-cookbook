#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

from common import (
    add_smoke_cli_args,
    bootstrap_training_test,
    build_sft_datum,
    smoke_config_from_args,
    smoke_step,
)

TEST_NAME = "sync smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synchronous MinT smoke test.")
    add_smoke_cli_args(parser)
    return parser.parse_args()


def main() -> int:
    from mint import types

    args = parse_args()
    ctx = bootstrap_training_test(TEST_NAME, config=smoke_config_from_args(args))
    datum = build_sft_datum(
        prompt="Question: What is 2 + 3?\nAnswer:",
        completion=" 5",
        tokenizer=ctx.tokenizer,
        types_module=types,
    )

    with smoke_step(TEST_NAME, "get training info"):
        info = ctx.training_client.get_info()
    model_name = getattr(getattr(info, "model_data", None), "model_name", None)
    if not model_name:
        raise RuntimeError("training_client.get_info() did not return model_data.model_name")
    print(f"{TEST_NAME}: info_model_name={model_name}")

    with smoke_step(TEST_NAME, "run forward"):
        forward = ctx.training_client.forward([datum], loss_fn="cross_entropy").result()
    if not getattr(forward, "loss_fn_outputs", None):
        raise RuntimeError("forward returned no loss outputs")

    with smoke_step(TEST_NAME, "run forward_backward"):
        fb = ctx.training_client.forward_backward([datum], loss_fn="cross_entropy").result()
    if not getattr(fb, "loss_fn_outputs", None):
        raise RuntimeError("forward_backward returned no loss outputs")

    with smoke_step(TEST_NAME, "run optim_step"):
        ctx.training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

    with smoke_step(TEST_NAME, "save weights and create sampling client"):
        sampler = ctx.training_client.save_weights_and_get_sampling_client(name=f"sync-sampler-{time.time_ns()}")
    with smoke_step(TEST_NAME, "get base model from in-process sampler"):
        sampler_base_model = sampler.get_base_model()
    if not sampler_base_model:
        raise RuntimeError("sampler.get_base_model() returned an empty model name")
    print(f"{TEST_NAME}: sampler_base_model={sampler_base_model}")

    prompt_tokens = ctx.tokenizer.encode("Question: What is 4 + 7?\nAnswer:", add_special_tokens=True)
    with smoke_step(TEST_NAME, "sample from in-process sampler"):
        sample = sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=12,
                temperature=0.0,
                stop_token_ids=[ctx.tokenizer.eos_token_id],
            ),
        ).result()
    if not getattr(sample, "sequences", None):
        raise RuntimeError("Sampling response did not contain any sequences")
    sampled_text = ctx.tokenizer.decode(sample.sequences[0].tokens).strip()
    print(f"{TEST_NAME}: sampled_text={sampled_text!r}")

    with smoke_step(TEST_NAME, "compute prompt logprobs"):
        prompt_logprobs = sampler.compute_logprobs(types.ModelInput.from_ints(tokens=prompt_tokens)).result()
    if prompt_logprobs is None:
        raise RuntimeError("compute_logprobs returned None")

    with smoke_step(TEST_NAME, "save weights for external sampler"):
        sampler_path = ctx.training_client.save_weights_for_sampler(
            name=f"sync-external-sampler-{time.time_ns()}"
        ).result().path
    print(f"{TEST_NAME}: sampler_path={sampler_path}")

    with smoke_step(TEST_NAME, "create external sampling client"):
        external_sampler = ctx.service_client.create_sampling_client(model_path=sampler_path)
    with smoke_step(TEST_NAME, "get base model from external sampler"):
        external_base_model = external_sampler.get_base_model()
    if not external_base_model:
        raise RuntimeError("external_sampler.get_base_model() returned an empty model name")
    print(f"{TEST_NAME}: external_base_model={external_base_model}")
    with smoke_step(TEST_NAME, "sample from external sampler"):
        external_sample = external_sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=12,
                temperature=0.0,
                stop_token_ids=[ctx.tokenizer.eos_token_id],
            ),
        ).result()
    if not getattr(external_sample, "sequences", None):
        raise RuntimeError("External sampling response did not contain any sequences")
    external_sampled_text = ctx.tokenizer.decode(external_sample.sequences[0].tokens).strip()
    print(f"{TEST_NAME}: external_sampled_text={external_sampled_text!r}")

    with smoke_step(TEST_NAME, "save training state"):
        state_path = ctx.training_client.save_state(name=f"sync-state-{time.time_ns()}").result().path
    print(f"{TEST_NAME}: state_path={state_path}")
    print(f"{TEST_NAME}: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{TEST_NAME}: FAIL: {exc}", file=sys.stderr)
        raise
