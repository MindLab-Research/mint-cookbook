import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SNAPSHOT_SCRIPT = REPO_ROOT / ".github" / "scripts" / "release_public_snapshot.sh"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True, env=env)


class ReleaseExportContractTest(unittest.TestCase):
    def create_repo(self, root: Path) -> None:
        run(["git", "init", "-b", "develop"], cwd=root)
        run(["git", "config", "user.name", "Test Bot"], cwd=root)
        run(["git", "config", "user.email", "test@example.com"], cwd=root)

    def run_release(
        self,
        source_repo: Path,
        alpha_repo: Path,
        public_repo: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        run(
            [
                "bash",
                str(RELEASE_SNAPSHOT_SCRIPT),
                "--source-repo-dir",
                str(source_repo),
                "--alpha-mirror-dir",
                str(alpha_repo),
                "--public-mirror-dir",
                str(public_repo),
                "--tag-name",
                "v0.0.0-test",
            ],
            cwd=REPO_ROOT,
            env=env,
        )

    def write_release_infra(self, source_repo: Path, include_body: str) -> None:
        (source_repo / ".github" / "scripts").mkdir(parents=True)
        (source_repo / ".github" / "workflows").mkdir(parents=True)
        (source_repo / ".github" / "public-include.txt").write_text(include_body, encoding="utf-8")
        script_copy = RELEASE_SNAPSHOT_SCRIPT.read_text(encoding="utf-8")
        (source_repo / ".github" / "scripts" / "release_public_snapshot.sh").write_text(
            script_copy,
            encoding="utf-8",
        )
        (source_repo / ".github" / "workflows" / "release.yml").write_text(
            "name: ignored\n",
            encoding="utf-8",
        )

    def test_release_skips_gitignored_paths_even_under_allowlisted_directories(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="mint-cookbook-release-export-"))
        try:
            source_repo = tmp_root / "source"
            alpha_repo = tmp_root / "alpha"
            public_repo = tmp_root / "public"
            for repo_dir in (source_repo, alpha_repo, public_repo):
                repo_dir.mkdir()
                self.create_repo(repo_dir)

            (source_repo / ".gitignore").write_text(
                "release/ignored-tracked.txt\nrelease/generated.tmp\n",
                encoding="utf-8",
            )
            (source_repo / "release").mkdir()
            (source_repo / "release" / "keep.txt").write_text("keep\n", encoding="utf-8")
            (source_repo / "release" / "ignored-tracked.txt").write_text("ignored\n", encoding="utf-8")
            (source_repo / "release" / "generated.tmp").write_text("generated\n", encoding="utf-8")
            self.write_release_infra(source_repo, "release\n")

            run(
                [
                    "git",
                    "add",
                    ".gitignore",
                    "release/keep.txt",
                    ".github/public-include.txt",
                    ".github/scripts/release_public_snapshot.sh",
                    ".github/workflows/release.yml",
                ],
                cwd=source_repo,
            )
            run(["git", "add", "-f", "release/ignored-tracked.txt"], cwd=source_repo)
            run(["git", "commit", "-m", "snapshot"], cwd=source_repo)
            run(["git", "commit", "--allow-empty", "-m", "alpha init"], cwd=alpha_repo)
            run(["git", "commit", "--allow-empty", "-m", "public init"], cwd=public_repo)

            self.run_release(source_repo, alpha_repo, public_repo)

            public_files = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", "main"],
                cwd=public_repo,
                text=True,
            ).splitlines()
            self.assertIn("release/keep.txt", public_files)
            self.assertNotIn("release/ignored-tracked.txt", public_files)
            self.assertNotIn("release/generated.tmp", public_files)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_release_ignores_repo_gitignore_but_not_runner_global_excludes(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="mint-cookbook-release-export-global-"))
        try:
            source_repo = tmp_root / "source"
            alpha_repo = tmp_root / "alpha"
            public_repo = tmp_root / "public"
            for repo_dir in (source_repo, alpha_repo, public_repo):
                repo_dir.mkdir()
                self.create_repo(repo_dir)

            (source_repo / ".gitignore").write_text("", encoding="utf-8")
            (source_repo / "release").mkdir()
            (source_repo / "release" / "keep.txt").write_text("keep\n", encoding="utf-8")
            self.write_release_infra(source_repo, "release\n")

            run(
                [
                    "git",
                    "add",
                    ".gitignore",
                    "release/keep.txt",
                    ".github/public-include.txt",
                    ".github/scripts/release_public_snapshot.sh",
                    ".github/workflows/release.yml",
                ],
                cwd=source_repo,
            )
            run(["git", "commit", "-m", "snapshot"], cwd=source_repo)
            run(["git", "commit", "--allow-empty", "-m", "alpha init"], cwd=alpha_repo)
            run(["git", "commit", "--allow-empty", "-m", "public init"], cwd=public_repo)

            global_home = tmp_root / "home"
            global_home.mkdir()
            global_gitconfig = tmp_root / "gitconfig"
            global_excludes = global_home / ".gitignore"
            global_excludes.write_text("release/keep.txt\n", encoding="utf-8")
            global_gitconfig.write_text(
                "[core]\n\texcludesFile = {path}\n".format(path=global_excludes),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["GIT_CONFIG_GLOBAL"] = str(global_gitconfig)
            self.run_release(source_repo, alpha_repo, public_repo, env=env)

            public_files = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", "main"],
                cwd=public_repo,
                text=True,
            ).splitlines()
            self.assertIn("release/keep.txt", public_files)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_alpha_snapshot_skips_gitignored_release_infra(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="mint-cookbook-release-snapshot-"))
        try:
            source_repo = tmp_root / "source"
            alpha_repo = tmp_root / "alpha"
            public_repo = tmp_root / "public"
            for repo_dir in (source_repo, alpha_repo, public_repo):
                repo_dir.mkdir()
                self.create_repo(repo_dir)

            (source_repo / ".gitignore").write_text(
                ".github/workflows/release.yml\n",
                encoding="utf-8",
            )
            (source_repo / "README.md").write_text("public\n", encoding="utf-8")
            (source_repo / ".env.example").write_text("MINT_API_KEY=sk-your-api-key-here\n", encoding="utf-8")
            (source_repo / "AGENTS.md").write_text("agents\n", encoding="utf-8")
            (source_repo / "tests").mkdir()
            (source_repo / "tests" / "README.md").write_text("tests\n", encoding="utf-8")
            self.write_release_infra(
                source_repo,
                ".env.example\n.gitignore\nAGENTS.md\nREADME.md\ntests\n",
            )

            run(
                [
                    "git",
                    "add",
                    ".gitignore",
                    ".env.example",
                    "AGENTS.md",
                    "README.md",
                    "tests/README.md",
                    ".github/public-include.txt",
                    ".github/scripts/release_public_snapshot.sh",
                ],
                cwd=source_repo,
            )
            run(["git", "add", "-f", ".github/workflows/release.yml"], cwd=source_repo)
            run(["git", "commit", "-m", "source snapshot"], cwd=source_repo)
            run(["git", "commit", "--allow-empty", "-m", "alpha init"], cwd=alpha_repo)
            run(["git", "commit", "--allow-empty", "-m", "public init"], cwd=public_repo)

            global_home = tmp_root / "runner-home"
            global_home.mkdir()
            global_gitconfig = tmp_root / "runner-gitconfig"
            global_excludes = global_home / ".gitignore"
            global_excludes.write_text("README.md\n", encoding="utf-8")
            global_gitconfig.write_text(
                "[core]\n\texcludesFile = {path}\n".format(path=global_excludes),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["GIT_CONFIG_GLOBAL"] = str(global_gitconfig)
            self.run_release(source_repo, alpha_repo, public_repo, env=env)

            alpha_files = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", "main"],
                cwd=alpha_repo,
                text=True,
            ).splitlines()
            public_files = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", "main"],
                cwd=public_repo,
                text=True,
            ).splitlines()
            self.assertIn(".github/public-include.txt", alpha_files)
            self.assertIn(".github/scripts/release_public_snapshot.sh", alpha_files)
            self.assertNotIn(".github/workflows/release.yml", alpha_files)
            self.assertIn("README.md", public_files)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
