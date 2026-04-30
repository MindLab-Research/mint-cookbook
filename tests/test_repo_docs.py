import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_REGISTRY = REPO_ROOT / "experiments" / "maintained.json"
ROOT_INDEX_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "experiments" / "README.md",
    REPO_ROOT / "docs" / "repo-overview.md",
)
MINT_RUNTIME_POLICY_DOCS = {
    REPO_ROOT / "README.md": (
        "that scaffold now defaults to `import mint`, `MINT_*`, and `--mint-timeout`",
    ),
    REPO_ROOT / "AGENTS.md": (
        "maintained `mint` runtime surface",
        "get the experiment working on `mint` first",
    ),
    REPO_ROOT / "experiments" / "README.md": (
        "New scaffold-derived experiments should use `import mint`, `MINT_BASE_URL`, `MINT_API_KEY`, and `--mint-timeout` by default",
    ),
    REPO_ROOT / "docs" / "repo-overview.md": (
        "新 scaffold 实验默认直接写 `import mint`、`MINT_BASE_URL`、`MINT_API_KEY` 和 `--mint-timeout`",
    ),
    REPO_ROOT / "scaffolds" / "README.md": (
        "The canonical scaffold now uses current `mint` runtime names inside `train.py`",
        "`--mint-timeout`",
    ),
    REPO_ROOT / "scaffolds" / "single_file_experiment" / "README.md": (
        "the scaffold code uses `mint` runtime names by default",
    ),
    REPO_ROOT / "scaffolds" / "single_file_experiment" / "naming.md": (
        "The canonical scaffold now uses `mint` runtime names by default",
        "`MINT_API_KEY` / `MINT_BASE_URL`",
        "`--mint-timeout`",
    ),
    REPO_ROOT / "skills" / "new-experiment" / "SKILL.md": (
        "new experiments should start directly on the maintained `mint` runtime surface",
        "`import mint`",
        "`MINT_API_KEY`",
        "`MINT_BASE_URL`",
    ),
    REPO_ROOT / "skills" / "new-experiment-plus" / "SKILL.md": (
        "The scaffold now defaults to `mint`",
    ),
}
MINT_TINKER_MIGRATION_GUIDE_DOCS = {
    REPO_ROOT / "experiments" / "README.md": (
        "import mint as tinker",
        "MINT_BASE_URL",
        "MINT_API_KEY",
        "migration bridge",
    ),
    REPO_ROOT / "docs" / "repo-overview.md": (
        "import mint as tinker",
        "MINT_BASE_URL",
        "MINT_API_KEY",
        "这只是迁移桥",
    ),
    REPO_ROOT / "scaffolds" / "README.md": (
        "import mint as tinker",
        "MINT_BASE_URL",
        "MINT_API_KEY",
        "should still import `mint` directly",
    ),
    REPO_ROOT / "scaffolds" / "single_file_experiment" / "naming.md": (
        "Legacy tinker-script bridge",
        "import mint as tinker",
        "MINT_BASE_URL",
        "MINT_API_KEY",
        "not as the default style",
    ),
    REPO_ROOT / "skills" / "mint-api" / "SKILL.md": (
        "import mint as tinker",
        "MINT_BASE_URL",
        "MINT_API_KEY",
        "do not use that alias as the default",
    ),
    REPO_ROOT / "skills" / "mint-api" / "mint_api_cheatsheet.md": (
        "import mint as tinker",
        "MINT_BASE_URL",
        "MINT_API_KEY",
        "migration bridge",
    ),
}
MINT_ENV_TEMPLATES = (
    REPO_ROOT / ".env.example",
    REPO_ROOT / "scaffolds" / "single_file_experiment" / "env.tpl",
    REPO_ROOT / "skills" / "project-harness-bootstrap" / "templates" / ".env.example.template",
)
FORBIDDEN_TINKER_RUNTIME_FRAGMENTS = (
    "TINKER_API_KEY",
    "TINKER_BASE_URL",
    "--tinker-timeout",
    "import tinker",
    "from tinker import types",
)
FORBIDDEN_TINKER_RUNTIME_DOC_FRAGMENTS = (
    "TINKER_API_KEY",
    "TINKER_BASE_URL",
    "--tinker-timeout",
)
LOCAL_AGENT_ROUTING_POLICY_DOCS = {
    REPO_ROOT / "README.md": (
        "skills/<name>/",
        ".codex/skills -> ../skills",
        ".claude/skills -> ../skills",
        "only checked-in source of truth",
    ),
    REPO_ROOT / "skills" / "README.md": (
        "`skills/` is the canonical home",
        ".codex/skills -> ../skills",
        ".claude/skills -> ../skills",
        "only checked-in source of truth",
    ),
}
FORBIDDEN_LOCAL_AGENT_DOC_FRAGMENTS = (
    "settings.local.json",
)
FORBIDDEN_PERSONAL_ENVIRONMENT_DOC_FRAGMENTS = (
    "mint-dev",
    "/root/code/lab/mint-cookbook",
    "/Users/leixiang/Desktop/mind/mint-cookbook",
    "<remote-host>",
    "<remote-repo-root>",
    "docs/remote-workflow.md",
    "local mirror -> sync -> remote run",
    "direct remote work",
    "authoritative remote repo root",
)
FORBIDDEN_INTERNAL_CLUSTER_PATH_FRAGMENTS = ("ve" "PFS-Mindverse",)
INTERNAL_PATH_SURFACE_FILES = (
    REPO_ROOT / "tests" / "common.py",
    REPO_ROOT / "skills" / "mint-api" / "mint_api_cheatsheet.md",
    REPO_ROOT / "experiments" / "dapo-aime" / "cache-env.sh",
    REPO_ROOT / "experiments" / "chat-dpo" / "tests" / "test_train.py",
    REPO_ROOT / "experiments" / "dapo-aime" / "tests" / "test_train.py",
    REPO_ROOT / "experiments" / "fingpt" / "tests" / "test_train.py",
    REPO_ROOT / "experiments" / "fingpt" / "data" / "make_sentiment_train_eval_subset.py",
    REPO_ROOT / "experiments" / "lawbench" / "data" / "eval" / "build_eval_manifest.py",
    REPO_ROOT / "experiments" / "lawbench" / "data" / "eval" / "build_train_eval_manifest.py",
    REPO_ROOT / "experiments" / "lawbench" / "data" / "eval" / "download_eval_raw.py",
    REPO_ROOT / "experiments" / "lawbench" / "data" / "train" / "build_train_manifest.py",
    REPO_ROOT / "experiments" / "lawbench" / "data" / "train" / "download_train_raw.py",
    REPO_ROOT / "experiments" / "lawbench" / "train.py",
    REPO_ROOT / "experiments" / "lawbench" / "tests" / "test_train.py",
    REPO_ROOT / "experiments" / "lawbench" / "tests" / "repro_mint_duplicate_prompt_error.py",
)
RESULT_STATUS_POLICY_DOCS = {
    REPO_ROOT / "AGENTS.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a maintained run is actually checked",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "experiments" / "README.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a maintained run is actually checked",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "docs" / "repo-overview.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "在有 checked run 之前先用",
        "只有在报告里的 run 确实已经 checked 时才切到",
    ),
}
SCAFFOLD_README_TEMPLATE = REPO_ROOT / "scaffolds" / "single_file_experiment" / "README.md.tpl"
RESULT_STATUS_GUIDE_DOCS = {
    REPO_ROOT / "scaffolds" / "README.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "scaffolds" / "single_file_experiment" / "README.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "scaffolds" / "profiles" / "eval.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the baseline is actually checked",
    ),
    REPO_ROOT / "scaffolds" / "profiles" / "sft.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "scaffolds" / "profiles" / "dpo.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "scaffolds" / "profiles" / "grpo.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "skills" / "new-experiment" / "SKILL.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "skills" / "new-experiment-plus" / "SKILL.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "skills" / "new-experiment-plus" / "references" / "upgrade_playbook.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only when the reported run is actually checked",
    ),
    REPO_ROOT / "skills" / "new-experiment-plus" / "references" / "file_upgrade_map.md": (
        "Status: \\`placeholder\\``",
        "Status: \\`checked\\``",
        "until a checked run exists",
        "only switches to `Status: \\`checked\\`` when the reported run is actually checked",
    ),
}


def load_maintained_experiments() -> list[dict[str, str]]:
    payload = json.loads(MAINTAINED_REGISTRY.read_text(encoding="utf-8"))
    return payload["maintained_experiments"]


def extract_maintained_block(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- maintained-experiments:start -->\n(.*?)\n<!-- maintained-experiments:end -->",
        text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing maintained experiment block in {path}")
    return [line.rstrip() for line in match.group(1).splitlines() if line.strip()]


def extract_section(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing ## {heading} section")
    return match.group(1)


def extract_shell_commands(text: str) -> list[str]:
    commands: list[str] = []
    for block in re.findall(r"```(?:[^\n]*)\n(.*?)```", text, re.DOTALL):
        current = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            current = f"{current} {line}".strip() if current else line
            if current.endswith("\\"):
                current = current[:-1].rstrip()
                continue
            commands.append(current)
            current = ""
        if current:
            commands.append(current)
    return commands


def extract_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class RepoDocsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maintained = load_maintained_experiments()

    def test_root_maintained_indexes_match_registry(self):
        expected = [
            f"- `{entry['name']}`: {entry['summary']}"
            for entry in self.maintained
        ]

        for path in ROOT_INDEX_DOCS:
            with self.subTest(path=path):
                self.assertEqual(extract_maintained_block(path), expected)

    def test_maintained_results_status_matches_registry(self):
        for entry in self.maintained:
            readme_path = REPO_ROOT / entry["path"] / "README.md"
            text = readme_path.read_text(encoding="utf-8")
            current_results = extract_section(text, "Current results")
            nonempty_lines = extract_nonempty_lines(current_results)
            first_nonempty = nonempty_lines[0] if nonempty_lines else None
            with self.subTest(readme=readme_path):
                self.assertIsNotNone(first_nonempty, msg=f"empty Current results section in {readme_path}")
                self.assertEqual(first_nonempty, f"Status: `{entry['results_status']}`")
                self.assertNotIn("TODO", current_results, msg=f"raw TODO placeholder leaked into {readme_path}")

                if entry["results_status"] == "placeholder":
                    self.assertIn(
                        "No maintained reportable run is checked in yet for this experiment.",
                        current_results,
                    )
                else:
                    self.assertNotIn(
                        "No maintained reportable run is checked in yet for this experiment.",
                        current_results,
                    )
                    self.assertGreaterEqual(
                        len(nonempty_lines),
                        3,
                        msg=f"checked Current results section should report more than a bare status line in {readme_path}",
                    )

    def test_repo_policy_docs_explain_status_transition(self):
        for path, expected_fragments in RESULT_STATUS_POLICY_DOCS.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn("Current results", text)
                for fragment in expected_fragments:
                    self.assertIn(fragment, text)

    def test_runtime_policy_docs_are_mint_first(self):
        for path, expected_fragments in MINT_RUNTIME_POLICY_DOCS.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for fragment in expected_fragments:
                    self.assertIn(fragment, text)
                for forbidden in FORBIDDEN_TINKER_RUNTIME_DOC_FRAGMENTS:
                    self.assertNotIn(forbidden, text)

    def test_legacy_tinker_migration_docs_keep_mint_alias_bridge_explicit(self):
        for path, expected_fragments in MINT_TINKER_MIGRATION_GUIDE_DOCS.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for fragment in expected_fragments:
                    self.assertIn(fragment, text)

    def test_maintained_train_code_stays_mint_first(self):
        required_fragments = (
            "import mint",
            "from mint import types",
            "MINT_BASE_URL",
            "MINT_API_KEY",
            "--mint-timeout",
        )
        for entry in self.maintained:
            train_path = REPO_ROOT / entry["path"] / "train.py"
            text = train_path.read_text(encoding="utf-8")
            with self.subTest(train_path=train_path):
                for fragment in required_fragments:
                    self.assertIn(fragment, text)
                for fragment in FORBIDDEN_TINKER_RUNTIME_FRAGMENTS:
                    self.assertNotIn(fragment, text)

    def test_maintained_checkpoint_rerun_docs_keep_base_model_explicit(self):
        for entry in self.maintained:
            experiment_root = REPO_ROOT / entry["path"]
            readme_text = (experiment_root / "README.md").read_text(encoding="utf-8")
            agents_text = (experiment_root / "AGENTS.md").read_text(encoding="utf-8")
            autoresearch_text = (experiment_root / "autoresearch.md").read_text(encoding="utf-8")
            quickstart = extract_section(readme_text, "Quickstart")
            recovery = extract_section(autoresearch_text, "Recovery and confirmation")

            with self.subTest(experiment=entry["name"], file="README.md"):
                self.assertIn("--base-model <sampler_path>", quickstart)

            with self.subTest(experiment=entry["name"], file="AGENTS.md"):
                self.assertIn("--base-model <sampler_path>", agents_text)

            with self.subTest(experiment=entry["name"], file="autoresearch.md"):
                self.assertIn("sampler_path", recovery)
                self.assertIn("--base-model", recovery)

            for file_name, text in (
                ("README.md", readme_text),
                ("AGENTS.md", agents_text),
                ("autoresearch.md", autoresearch_text),
            ):
                for command in extract_shell_commands(text):
                    with self.subTest(experiment=entry["name"], file=file_name, command=command):
                        self.assertFalse(
                            "--eval-only" in command and "--load-checkpoint-path" in command,
                            msg=f"eval-only checkpoint confirmation must use sampler_path, not load-checkpoint-path: {command}",
                        )

    def test_scaffold_readme_template_uses_status_placeholder_in_current_results(self):
        text = SCAFFOLD_README_TEMPLATE.read_text(encoding="utf-8")
        current_results = extract_section(text, "Current results")
        first_nonempty = next(
            (line.strip() for line in current_results.splitlines() if line.strip()),
            None,
        )

        self.assertEqual(first_nonempty, "Status: `placeholder`")
        self.assertIn("Do not leave a raw `TODO` placeholder here.", current_results)
        self.assertEqual(current_results.count("TODO"), 1)
        self.assertIn("switch it to `Status: \\`checked\\`` only when", current_results)

    def test_status_transition_guidance_is_present_in_related_scaffold_docs(self):
        for path, expected_fragments in RESULT_STATUS_GUIDE_DOCS.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn("Current results", text)
                for fragment in expected_fragments:
                    self.assertIn(fragment, text)

    def test_scaffold_runtime_template_uses_mint_names(self):
        text = (REPO_ROOT / "scaffolds" / "single_file_experiment" / "train.py.tpl").read_text(
            encoding="utf-8"
        )

        for fragment in (
            'LOCAL_ENV_KEYS = {"MINT_API_KEY", "MINT_BASE_URL"}',
            "import mint",
            "from mint import types",
            'base_url=os.environ.get("MINT_BASE_URL")',
            "--mint-timeout",
            'os.environ.get("MINT_API_KEY")',
            "Missing MINT_API_KEY",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

        for fragment in FORBIDDEN_TINKER_RUNTIME_FRAGMENTS:
            with self.subTest(forbidden=fragment):
                self.assertNotIn(fragment, text)

    def test_env_templates_only_expose_mint_runtime_keys(self):
        for path in MINT_ENV_TEMPLATES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertIn("MINT_BASE_URL", text)
                self.assertIn("MINT_API_KEY", text)
                self.assertNotIn("TINKER_BASE_URL", text)
                self.assertNotIn("TINKER_API_KEY", text)

    def test_repo_docs_keep_local_agent_routing_out_of_repo_contract(self):
        for path, expected_fragments in LOCAL_AGENT_ROUTING_POLICY_DOCS.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for fragment in expected_fragments:
                    self.assertIn(fragment, text)
                for fragment in FORBIDDEN_LOCAL_AGENT_DOC_FRAGMENTS:
                    self.assertNotIn(fragment, text)

    def test_open_source_docs_avoid_personal_execution_topology(self):
        active_repo_docs = (
            REPO_ROOT / "README.md",
            REPO_ROOT / "AGENTS.md",
            REPO_ROOT / "docs" / "repo-overview.md",
            REPO_ROOT / "experiments" / "README.md",
            REPO_ROOT / "tests" / "README.md",
            REPO_ROOT / "skills" / "README.md",
        )
        for path in active_repo_docs:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for fragment in FORBIDDEN_PERSONAL_ENVIRONMENT_DOC_FRAGMENTS:
                    self.assertNotIn(fragment, text)

        self.assertFalse((REPO_ROOT / "docs" / "remote-workflow.md").exists())

        for entry in self.maintained:
            for relative in ("README.md", "AGENTS.md"):
                path = REPO_ROOT / entry["path"] / relative
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=path):
                    for fragment in FORBIDDEN_PERSONAL_ENVIRONMENT_DOC_FRAGMENTS:
                        self.assertNotIn(fragment, text)

    def test_maintained_registry_does_not_store_personal_remote_workdirs(self):
        payload = json.loads(MAINTAINED_REGISTRY.read_text(encoding="utf-8"))
        for entry in payload["maintained_experiments"]:
            with self.subTest(experiment=entry["name"]):
                self.assertNotIn("remote_workdir", entry)

    def test_experiment_data_contract_files_avoid_personal_remote_paths(self):
        for entry in self.maintained:
            experiment_root = REPO_ROOT / entry["path"]
            for relative in ("data/README.md", "data/sources.yaml"):
                path = experiment_root / relative
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=path):
                    for fragment in FORBIDDEN_PERSONAL_ENVIRONMENT_DOC_FRAGMENTS:
                        self.assertNotIn(fragment, text)

    def test_active_runtime_files_avoid_internal_cluster_paths(self):
        for path in INTERNAL_PATH_SURFACE_FILES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                for fragment in FORBIDDEN_INTERNAL_CLUSTER_PATH_FRAGMENTS:
                    self.assertNotIn(fragment, text)

    def test_scaffold_readme_template_data_section_avoids_raw_todo(self):
        text = SCAFFOLD_README_TEMPLATE.read_text(encoding="utf-8")
        data_section = extract_section(text, "Data")

        self.assertNotIn("replace with `TODO`", data_section)
        self.assertIn("raw `TODO` markers", data_section)
        self.assertEqual(data_section.count("TODO"), 1)

    def test_scaffold_readme_template_keeps_canonical_section_order(self):
        text = SCAFFOLD_README_TEMPLATE.read_text(encoding="utf-8")
        headings = [
            "## Quickstart",
            "## Fast contract tests",
            "## Live smoke tests",
            "## Data",
            "## Current results",
            "## References",
        ]

        positions = [text.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
