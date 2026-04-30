import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SCRIPT = REPO_ROOT / "skills" / "project-harness-bootstrap" / "scripts" / "bootstrap_harness.py"
SOURCE_SCAFFOLDS = REPO_ROOT / "scaffolds"


class HarnessBootstrapTest(unittest.TestCase):
    def run_bootstrap(self, root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(BOOTSTRAP_SCRIPT),
                "--root",
                str(root),
                "--project",
                "mint-bootstrap-check",
                *extra_args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_bootstrap_harness_materializes_current_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "mint-bootstrap-check"
            transient_ds_store = SOURCE_SCAFFOLDS / ".DS_Store"
            transient_pyc = SOURCE_SCAFFOLDS / "single_file_experiment" / "transient_filter_check.pyc"
            original_ds_store = transient_ds_store.read_bytes() if transient_ds_store.exists() else None
            original_pyc = transient_pyc.read_bytes() if transient_pyc.exists() else None
            transient_ds_store.write_bytes(b"transient-ds-store\n")
            transient_pyc.write_bytes(b"transient-pyc")
            try:
                result = self.run_bootstrap(root)
            finally:
                if original_ds_store is None:
                    transient_ds_store.unlink(missing_ok=True)
                else:
                    transient_ds_store.write_bytes(original_ds_store)
                if original_pyc is None:
                    transient_pyc.unlink(missing_ok=True)
                else:
                    transient_pyc.write_bytes(original_pyc)

            self.assertEqual(result.returncode, 0, msg=result.stderr)

            self.assertTrue((root / "README.md").exists())
            self.assertTrue((root / "AGENTS.md").exists())
            self.assertTrue((root / ".env.example").exists())
            self.assertTrue((root / ".gitignore").exists())
            self.assertTrue((root / "experiments" / "README.md").exists())
            self.assertTrue((root / "experiments" / ".gitkeep").exists())
            self.assertTrue((root / "scaffolds" / "README.md").exists())
            self.assertTrue((root / "scaffolds" / "lifecycle.md").exists())
            self.assertEqual(list((root / "scaffolds").rglob("__pycache__")), [])
            self.assertEqual(list((root / "scaffolds").rglob("*.pyc")), [])
            self.assertEqual(list((root / "scaffolds").rglob(".DS_Store")), [])

            readme = (root / "README.md").read_text(encoding="utf-8")
            self.assertIn("`.env.example` - shared checked-in live env template", readme)
            self.assertIn("`skills/` - optional repo-local skills", readme)
            self.assertIn(
                "uv run train.py --eval-only --eval-data <full_eval_path>",
                readme,
            )
            self.assertIn("autoresearch.sh", readme)

            agents = (root / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn(
                "canonical benchmark confirmation entrypoint",
                agents,
            )
            self.assertIn("When this repo also carries Codex skills", agents)

            env_template = (root / ".env.example").read_text(encoding="utf-8")
            self.assertIn("MINT_BASE_URL=", env_template)

            self.assertIn(
                "start from the maintained `mint` runtime surface",
                readme,
            )

            experiments_readme = (root / "experiments" / "README.md").read_text(encoding="utf-8")
            self.assertIn(
                "`uv run train.py --eval-only --eval-data <full_eval_path>` should remain the benchmark confirmation entrypoint.",
                experiments_readme,
            )
            self.assertIn(
                "`autoresearch.sh` should be the automation wrapper that `pi-autoresearch` runs around that path.",
                experiments_readme,
            )
            self.assertIn(
                "`autoresearch.md` should define the wrapper protocol and benchmark scope, not serve as a general session log.",
                experiments_readme,
            )

            self.assertIn(".env.example", result.stdout)
            self.assertIn(".gitignore", result.stdout)
            self.assertIn("benchmark confirmation through", result.stdout)
            self.assertNotIn("__pycache__", result.stdout)
            self.assertNotIn(".DS_Store", result.stdout)

    def test_bootstrap_harness_retrofit_keeps_existing_scaffolds_and_backfills_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "mint-bootstrap-check"
            first = self.run_bootstrap(root)
            self.assertEqual(first.returncode, 0, msg=first.stderr)

            local_root_readme = root / "README.md"
            local_root_readme.write_text("local root note\n", encoding="utf-8")
            local_scaffold_readme = root / "scaffolds" / "README.md"
            local_scaffold_readme.write_text("local retrofit note\n", encoding="utf-8")
            missing_lifecycle = root / "scaffolds" / "lifecycle.md"
            missing_lifecycle.unlink()

            result = self.run_bootstrap(root, "--mode", "retrofit", "--overwrite")
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            self.assertIn("# mint-bootstrap-check", local_root_readme.read_text(encoding="utf-8"))
            self.assertEqual(local_scaffold_readme.read_text(encoding="utf-8"), "local retrofit note\n")
            self.assertTrue(missing_lifecycle.exists())
            self.assertIn("keep", result.stdout)
            self.assertIn("write", result.stdout)
            self.assertIn("/README.md", result.stdout)
            self.assertIn("scaffolds/README.md", result.stdout)
            self.assertIn("scaffolds/lifecycle.md", result.stdout)

    def test_bootstrap_harness_help_stays_narrow(self):
        result = subprocess.run(
            [sys.executable, str(BOOTSTRAP_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--mode {greenfield,retrofit}", result.stdout)
        self.assertNotIn("--stack", result.stdout)
        self.assertNotIn("Primary stack hint", result.stdout)
        normalized_stdout = " ".join(result.stdout.split())
        self.assertIn(
            "retrofit mode still keeps existing scaffold files",
            normalized_stdout,
        )


if __name__ == "__main__":
    unittest.main()
