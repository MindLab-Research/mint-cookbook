# Repo Verification Tests

This directory holds two repo-level verification layers:

- fast local contract tests that check scaffold and harness behavior without calling live APIs
- live MinT smoke scripts that keep the local skills aligned with the actual SDK behavior

These are support checks for the repo, not part of the `experiments/` contract and not a substitute for experiment-local documentation.

## Files

- `sync_smoke.py` - sync preflight plus `get_info`, `forward`, SFT step, sampling, sampler `get_base_model`, checkpoint save, `create_sampling_client`, `compute_logprobs`
- `checkpoint_resume_smoke.py` - `save_state`, then a fresh `create_lora_training_client(...)` plus `load_state_with_optimizer(...)`, followed by another train step and sampler probe
- `loss_smoke.py` - combined loss smoke covering minimal `importance_sampling` and `ppo` RL paths plus `forward_backward_custom` with a pairwise preference loss; the `importance_sampling` and `ppo` paths also save state and resume through the supported MinT flow
- `async_smoke.py` - async client creation plus the current SDK await pattern for `get_info_async`, `forward_async`, `forward_backward_async`, `optim_step_async`, `sample_async`, `compute_logprobs_async`, and `create_sampling_client_async`, written mostly inline with `smoke_step_async`
- `run_all.sh` - runs all live smoke tests in sequence
- `.env` - local secrets for these tests only, gitignored
- `common.py` - shared smoke-test bootstrap plus only a few repeated multi-step helpers (`load_env`, `preflight_connection`, tokenizer/model selection, `build_sft_datum`, `smoke_step`, `smoke_step_async`, resume flow); the shared resume helper intentionally uses the currently supported MinT pattern: fresh LoRA client plus `load_state_with_optimizer(...)`

## Setup

The scripts load `tests/.env` to fill in missing variables, but they do not overwrite values that are already present in the shell environment.
Like `experiments/dapo-aime`, `tests/.env` is only used for MinT connection settings:

```bash
MINT_BASE_URL=https://mint-cn.macaron.xin
MINT_API_KEY=...
```

All other smoke-test knobs now stay out of `tests/.env`.
`tests/` is its own small `uv` project because the smoke scripts import `mint`, `transformers`, and `torch` directly.
If the host does not already have Python 3.11+ on the search path, the first `uv run` may bootstrap a managed interpreter for this directory.
The scripts default to `--base-model Qwen/Qwen3-0.6B --lora-rank 4 --timeout-seconds 600`, so the common path needs no extra flags.
If you want to override them, pass `--base-model`, `--lora-rank`, or `--timeout-seconds` on the command line.
Smoke tests default `--lora-rank` to `4`; the scaffold baseline for new experiments defaults to `16`, which is intentional because the tests optimize for cheap smoke coverage while experiment baselines optimize for more typical training defaults.

Fast local contract tests that do not call MinT live APIs live here too:

- `test_scaffold_template.py` - checks scaffold artifact and append-stream metadata behavior
- `test_harness_bootstrap.py` - checks `project-harness-bootstrap` can materialize a fresh repo with the current harness contract
- `test_repo_docs.py` - checks maintained registry mirrors, repo and scaffold `Current results` status wording, mint-first scaffold runtime policy, maintained checkpoint-rerun doc wording, and that open-source docs stay free of personal execution-topology or repo-local agent-config contract

## Run

```bash
cd tests
bash run_all.sh
```

`run_all.sh` forwards its arguments to every smoke script when you need an override:

```bash
cd tests
bash run_all.sh --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 --lora-rank 8 --timeout-seconds 900
```

Or run a single script:

```bash
cd tests
uv run python sync_smoke.py
```

For the local contract tests, use plain `python` from the repo root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

If you want experiment-level live coverage, use each experiment's local suite instead of these repo support checks:

- `chat-dpo`: `cd experiments/chat-dpo && uv run python -m unittest tests.test_train`
- `fingpt`: `cd experiments/fingpt && uv run python -m unittest tests.test_train`
- `lawbench`: `cd experiments/lawbench && uv run python -m unittest tests.test_train`
- `dapo-aime`: `cd experiments/dapo-aime && uv run python -m unittest tests.test_train tests.test_async_compat tests.test_rl_logprobs`

If you only need the cheapest real eval-only remote check before a bigger benchmark rerun, the single-test entrypoints are:

- `dapo-aime`: `cd experiments/dapo-aime && uv run python -m unittest tests.test_train.LiveDAPOAIMEFlowTest.test_eval_only_live_smoke`
- `fingpt`: `cd experiments/fingpt && uv run python -m unittest tests.test_train.LiveFinGPTFlowTest.test_eval_only_live_smoke`
- `lawbench`: `cd experiments/lawbench && uv run python -m unittest tests.test_train.LiveLawBenchFlowTest.test_eval_only_live_smoke`

## Notes

- These are real remote API calls, not mocks.
- They use very small payloads to keep runtime and cost low.
- The scripts print the chosen model, step timing, sampled text, and checkpoint paths so you can compare them with the shared skills under `skills/`.
- The shared tokenizer helper prefers a cached Hugging Face snapshot path over `training_client.get_tokenizer()` because loading by model ID can still trigger Hub metadata requests inside `transformers`.

## Resume support

Current MinT behavior on this repo's endpoint is:

- Supported for true training resume: `save_state(...)`, then create a fresh `create_lora_training_client(...)`, then call `load_state_with_optimizer(...)` on that new client.
- Supported for sampler/eval follow-ups: save sampler weights and create a sampling client from that sampler path, or restore a fresh training client with `load_state_with_optimizer(...)` and then save sampler weights.
- Not supported in this repo as a general resume recipe: `service_client.create_training_client_from_state(...)`. The current SDK helper first calls `/api/v1/weights_info`, which returns `404 Not Found` on the active server.
- Not supported as a stable live-client resume path: calling `training_client.load_state(...)` on an already-running Megatron LoRA training client. In local smoke runs this hit CUDA illegal memory access during the explicit-load adapter snapshot path.
- Not a true resume path even when it appears to work: fresh `create_lora_training_client(...)` plus `load_state(...)`. That restores weights only, not optimizer state, so it should be treated as weight reload rather than training continuation.

When adding new examples, tests, or experiment entrypoints, prefer the supported recipe above and document unsupported paths explicitly.

## Coverage

- The default smoke suite covers sync SFT, async SFT, checkpoint resume through `create_lora_training_client(...) + load_state_with_optimizer(...)`, and combined loss coverage for `importance_sampling`, `ppo` and `forward_backward_custom`.

## Historical timing (Qwen3-4B-Instruct-2507, LoRA rank 4)

Measured 2026-04-20. Times vary with server load.

| Test | Wall time | Slowest steps |
|------|-----------|---------------|
| `sync_smoke.py` | ~2 min 42 s | `optim_step` 28 s, `save_weights_for_sampler` 24 s, `save_weights + create_sampling_client` 24 s, `forward` 22 s, `create_lora_training_client` 18 s, `forward_backward` 17 s |
| `async_smoke.py` | ~2 min 15 s | `save_weights + create_sampling_client` 27 s, `forward_backward_async` 25 s, `save_weights_for_sampler_async` 24 s, `create_lora_training_client` 24 s, `save_state` 10 s |
| `checkpoint_resume_smoke.py` | ~3 min 7 s | `resumed optim_step` 34 s, `create resumed training_client` 27 s, `save resumed weights + sampling_client` 27 s, `optim_step` 24 s, `forward_backward` 20 s, `resumed forward_backward` 15 s, `create_lora_training_client` 12 s, `load_state_with_optimizer` 12 s |
| `loss_smoke.py` | ~7 min 20 s | `forward_backward_custom` 56 s, `ppo resumed optim_step` 33 s, `save weights + sampling_client` (×2) 24–32 s, `resumed forward_backward` (×2) 29–32 s, `optim_step` (×3) 20–29 s, `create resumed training_client` (×2) 16–25 s |
| **Total (`run_all.sh`)** | **~15–16 min** | |

> `create_lora_training_client` 10–25 s and `save_weights*` 24–32 s are consistently the heaviest single RPCs. The `loss_smoke` test is the longest because it runs two full save→resume→retrain cycles (importance_sampling + ppo) plus a custom-loss forward_backward.
