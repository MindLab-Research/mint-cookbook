# Harness principles for AI-first repos

This skill is inspired by OpenAI's Harness Engineering article and the Codex execution-plan workflow.

Primary references:

- https://openai.com/index/harness-engineering/
- https://cookbook.openai.com/articles/codex_exec_plans

## What matters in practice

1. The repo must be the system of record.
   - Architecture, plans, reliability notes, and operating constraints should live in versioned files.
   - Stable knowledge should not remain trapped in chat history.

2. `AGENTS.md` should stay short.
   - Use it as a router, not as a giant wall of policy text.
   - It should point to the deeper files agents actually need.

3. Plans are first-class artifacts.
   - For anything meaningfully multi-step or multi-hour, create an execution plan before implementation.
   - Keep active and completed plans separate so the repo stays navigable.

4. Artifacts and evidence need stable homes.
   - Screenshots, logs, traces, and run outputs should land in predictable directories.
   - This makes debugging, reviews, and AI handoffs much easier.
   - When runs emit **machine-readable** summaries (for example `analysis_manifest.json`) or **stable stdout markers** (for example `METRIC`, `ANALYSIS_MANIFEST`), document them in the experiment `README.md` and keep **repo harness docs** (`AGENTS.md`, `scaffolds/README.md`, Codex skills) in sync in the **same commit** so the next agent does not reverse-engineer contracts from code alone. See `AGENTS.md` → *When you change logging, artifacts, or stdout contracts*.

5. Quality, reliability, and security need explicit docs.
   - If these expectations are only implicit, agents will drift.
   - Put the desired invariants in files that can be updated with the codebase.

6. New sessions need a fast path back into the repo.
   - Keep a short current-focus record and a short session-start prompt in the repo.
   - The next agent should know what to read first without reverse-engineering the whole tree.

7. Separate durable knowledge from transient handoffs.
   - Feature docs, workflow docs, and work records serve different jobs.
   - This keeps the repo searchable without turning it into a single giant scratchpad.

8. Retrofit beats reinvention.
   - In existing repos, add the harness around working code instead of forcing a total re-layout.
   - The goal is better operating leverage, not bureaucracy.

## Expanded harness file set for mature repos

This is not the bootstrap baseline for this repo skill. Treat it as an example of how a minimal harness may grow once the repo genuinely needs more operating structure.

A practical expanded set is:

- `AGENTS.md`
- `PLANS.md`
- `ARCHITECTURE.md`
- `docs/exec-plans/active/`
- `docs/exec-plans/completed/`
- `docs/features/`
- `docs/flows/`
- `docs/records/WORK_RECORDS.md`
- `docs/records/sessions/`
- `docs/prompts/NEW_SESSION.md`
- `docs/QUALITY_SCORE.md`
- `docs/RELIABILITY.md`
- `docs/SECURITY.md`
- `artifacts/screenshots/`
- `artifacts/runs/`

## Skill pairing guidance

This bootstrap skill is the foundation, not the whole operating system. Pair it with:

- `doc` for repo-local documentation work
- `playwright` and `playwright-interactive` for browser QA and product smoke tests
- `screenshot` for visual evidence
- `pdf`, `transcribe`, or `spreadsheet` when important inputs arrive in those formats
