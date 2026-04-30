# Shared Skills

`skills/` is the canonical home for repo-local agent skills.

## How To Use

1. Add or update a skill under `skills/<name>/`.
2. Keep the skill entrypoint in `skills/<name>/SKILL.md`.
3. Keep any supporting `references/`, `scripts/`, `templates/`, or `agents/` directories inside that same skill directory.
4. For local discovery, symlink tool-specific skill directories back to this source of truth, for example `.codex/skills -> ../skills` and `.claude/skills -> ../skills`.
5. Do not duplicate skill contents under `.codex/` or `.claude/`; `skills/` remains the only checked-in source of truth.
