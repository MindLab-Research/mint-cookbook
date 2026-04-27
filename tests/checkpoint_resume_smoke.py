#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

from common import (
    add_smoke_cli_args,
    bootstrap_training_test,
    build_sft_datum,
    get_tokenizer,
    resume_training_client_from_state,
    smoke_config_from_args,
    smoke_step,
)

TEST_NAME = "checkpoint resume smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MinT checkpoint resume smoke test.")
    add_smoke_cli_args(parser)
    return parser.parse_args()


def main() -> int:
    from mint import types

    args = parse_args()
    config = smoke_config_from_args(args)
    ctx = bootstrap_training_test(TEST_NAME, config=config)
    datum = build_sft_datum(
        prompt="Question: What is 6 + 9?\nAnswer:",
        completion=" 15",
        tokenizer=ctx.tokenizer,
        types_module=types,
    )

    with smoke_step(TEST_NAME, "run forward_backward"):
        ctx.training_client.forward_backward([datum], loss_fn="cross_entropy").result()
    with smoke_step(TEST_NAME, "run optim_step"):
        ctx.training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

    with smoke_step(TEST_NAME, "save training state"):
        state_path = ctx.training_client.save_state(name=f"resume-state-{time.time_ns()}").result().path
    print(f"{TEST_NAME}: state_path={state_path}")

    resumed_client, resume_mode = resume_training_client_from_state(
        TEST_NAME,
        ctx.service_client,
        ctx.model,
        state_path,
        config=config,
    )
    with smoke_step(TEST_NAME, f"load tokenizer for {ctx.model}"):
        resumed_tokenizer = get_tokenizer(resumed_client, ctx.model)
    with smoke_step(TEST_NAME, "run resumed forward_backward"):
        resumed_client.forward_backward([datum], loss_fn="cross_entropy").result()
    with smoke_step(TEST_NAME, "run resumed optim_step"):
        resumed_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()

    with smoke_step(TEST_NAME, "save resumed weights and create sampling client"):
        resumed_sampler = resumed_client.save_weights_and_get_sampling_client(name=f"resume-sampler-{time.time_ns()}")
    prompt_tokens = resumed_tokenizer.encode("Question: What is 3 + 8?\nAnswer:", add_special_tokens=True)
    with smoke_step(TEST_NAME, "sample from resumed sampler"):
        sample = resumed_sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=12,
                temperature=0.0,
                stop_token_ids=[resumed_tokenizer.eos_token_id],
            ),
        ).result()
    if not getattr(sample, "sequences", None):
        raise RuntimeError("Sampling response did not contain any sequences")
    resumed_sampled_text = resumed_tokenizer.decode(sample.sequences[0].tokens).strip()
    print(f"{TEST_NAME}: resume_mode={resume_mode}")
    print(f"{TEST_NAME}: resumed_sampled_text={resumed_sampled_text!r}")
    print(f"{TEST_NAME}: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{TEST_NAME}: FAIL: {exc}", file=sys.stderr)
        raise
