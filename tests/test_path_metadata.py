import importlib.machinery
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CASES = (
    (
        "lawbench_build_eval_manifest",
        REPO_ROOT / "experiments" / "lawbench" / "data" / "eval" / "build_eval_manifest.py",
        REPO_ROOT / "experiments" / "lawbench",
        Path("data/eval/full.jsonl"),
    ),
    (
        "lawbench_build_train_eval_manifest",
        REPO_ROOT / "experiments" / "lawbench" / "data" / "eval" / "build_train_eval_manifest.py",
        REPO_ROOT / "experiments" / "lawbench",
        Path("data/eval/train_eval_200.jsonl"),
    ),
    (
        "lawbench_download_eval_raw",
        REPO_ROOT / "experiments" / "lawbench" / "data" / "eval" / "download_eval_raw.py",
        REPO_ROOT / "experiments" / "lawbench",
        Path("data/eval/raw/lawbench-official/1-1.json"),
    ),
    (
        "lawbench_build_train_manifest",
        REPO_ROOT / "experiments" / "lawbench" / "data" / "train" / "build_train_manifest.py",
        REPO_ROOT / "experiments" / "lawbench",
        Path("data/train/full.jsonl"),
    ),
    (
        "lawbench_download_train_raw",
        REPO_ROOT / "experiments" / "lawbench" / "data" / "train" / "download_train_raw.py",
        REPO_ROOT / "experiments" / "lawbench",
        Path("data/train/raw/disc-law-sft/DISC-Law-SFT-Pair.jsonl"),
    ),
    (
        "fingpt_make_sentiment_train_eval_subset",
        REPO_ROOT / "experiments" / "fingpt" / "data" / "make_sentiment_train_eval_subset.py",
        REPO_ROOT / "experiments" / "fingpt",
        Path("data/benchmarks/sentiment/train-eval-160/all/test.jsonl"),
    ),
)


def load_module(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class PortablePathContractTest(unittest.TestCase):
    def test_portable_path_relativizes_experiment_local_paths(self):
        for module_name, path, experiment_dir, relative_path in SCRIPT_CASES:
            module = load_module(module_name, path)
            with self.subTest(script=path):
                self.assertEqual(module.portable_path(relative_path), relative_path.as_posix())
                self.assertEqual(
                    module.portable_path(experiment_dir / relative_path),
                    relative_path.as_posix(),
                )

    def test_portable_path_preserves_outside_absolute_paths(self):
        outside_path = Path("/tmp/mint-cookbook-portable-path-check")
        for module_name, path, _experiment_dir, _relative_path in SCRIPT_CASES:
            module = load_module(module_name, path)
            with self.subTest(script=path):
                self.assertEqual(module.portable_path(outside_path), str(outside_path))

    def test_resolve_experiment_path_accepts_repo_root_relative_paths(self):
        for module_name, path, experiment_dir, relative_path in SCRIPT_CASES:
            module = load_module(module_name, path)
            repo_relative = experiment_dir.relative_to(REPO_ROOT) / relative_path
            with self.subTest(script=path):
                self.assertEqual(
                    module.resolve_experiment_path(repo_relative),
                    REPO_ROOT / repo_relative,
                )

    def test_chat_dpo_live_env_overrides_stale_cache_vars(self):
        module = load_module(
            "chat_dpo_live_test_train",
            REPO_ROOT / "experiments" / "chat-dpo" / "tests" / "test_train.py",
        )
        stale_cache = "/dev/null/not-a-cache-dir"
        original_env_path = module.ENV_PATH
        module.ENV_PATH = Path(tempfile.gettempdir()) / "chat-dpo-no-env-file"
        try:
            with mock.patch.dict(
                module.os.environ,
                {
                    "HF_HOME": stale_cache,
                    "HUGGINGFACE_HUB_CACHE": stale_cache,
                    "TRANSFORMERS_CACHE": stale_cache,
                },
                clear=True,
            ):
                env = module.load_live_env()
        finally:
            module.ENV_PATH = original_env_path

        self.assertNotEqual(env["HF_HOME"], stale_cache)
        self.assertNotEqual(env["HUGGINGFACE_HUB_CACHE"], stale_cache)
        self.assertNotEqual(env["TRANSFORMERS_CACHE"], stale_cache)


if __name__ == "__main__":
    unittest.main()
