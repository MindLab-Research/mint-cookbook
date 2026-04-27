import asyncio
import argparse
import json
import importlib.machinery
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "scaffolds" / "single_file_experiment" / "train.py.tpl"


def load_template_module():
    loader = importlib.machinery.SourceFileLoader("scaffold_train_template", str(TEMPLATE_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class _FakeTrainingClient:
    def forward_backward(self, datums, loss_fn):
        self.last_forward_backward = {"datums": datums, "loss_fn": loss_fn}
        return {"loss_fn_outputs": [{"logprobs": [-1.0]} for _ in datums]}

    def optim_step(self, params):
        self.last_optim_params = params
        return types.SimpleNamespace(metrics={"step_time_seconds": 0.123})

    def save_weights_and_get_sampling_client(self, name="eval"):
        self.last_sampling_name = name
        return types.SimpleNamespace(model_path="tinker://sampler_weights/fake")


class ScaffoldTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = load_template_module()

    def make_args(self, **overrides):
        defaults = {
            "base_model": "Qwen/Qwen3-0.6B",
            "seed": 1,
            "rank": 8,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "max_steps": 1,
            "batch_size": 1,
            "lr_schedule": "linear",
            "eval_every": 0,
            "save_every": 0,
            "train_metrics_every": 0,
            "train_print_every": 0,
            "load_checkpoint_path": "",
            "max_concurrent_requests": 1,
            "eval_temperature": 0.0,
            "eval_max_tokens": 32,
            "eval_top_p": 1.0,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_reset_supervised_append_streams_removes_disabled_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            targets = [
                log_path / "eval" / "periodic.jsonl",
                log_path / "train" / "checkpoints.jsonl",
                log_path / "train" / "metrics.jsonl",
                log_path / "train" / "batches.jsonl",
            ]
            for path in targets:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stale\n", encoding="utf-8")

            self.template.reset_supervised_append_streams(
                log_path,
                include_periodic_eval=False,
                include_checkpoints=True,
                include_train_metrics=False,
                include_batch_trace=False,
            )

            self.assertFalse((log_path / "eval" / "periodic.jsonl").exists())
            self.assertTrue((log_path / "train" / "checkpoints.jsonl").exists())
            self.assertEqual(
                (log_path / "train" / "checkpoints.jsonl").read_text(encoding="utf-8"),
                "",
            )
            self.assertFalse((log_path / "train" / "metrics.jsonl").exists())
            self.assertFalse((log_path / "train" / "batches.jsonl").exists())

    def test_reset_supervised_append_streams_preserves_files_on_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            checkpoint_path = log_path / "train" / "checkpoints.jsonl"
            metrics_path = log_path / "train" / "metrics.jsonl"
            for path in (checkpoint_path, metrics_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep\n", encoding="utf-8")

            self.template.reset_supervised_append_streams(
                log_path,
                resume_checkpoint={"state_path": "mint://resume/state"},
                include_periodic_eval=False,
                include_checkpoints=False,
                include_train_metrics=False,
                include_batch_trace=False,
            )

            self.assertEqual(checkpoint_path.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(metrics_path.read_text(encoding="utf-8"), "keep\n")

    def test_write_outputs_rebuilds_artifacts_after_stale_stream_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            periodic_path = log_path / "eval" / "periodic.jsonl"
            checkpoints_path = log_path / "train" / "checkpoints.jsonl"
            metrics_path = log_path / "train" / "metrics.jsonl"
            batches_path = log_path / "train" / "batches.jsonl"
            for path in (periodic_path, checkpoints_path, metrics_path, batches_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"stale": true}\n', encoding="utf-8")

            self.template.write_json(
                log_path / "run.json",
                {
                    "experiment": "scaffold",
                    "artifacts": {
                        "eval": {"periodic": str(periodic_path)},
                        "train": {
                            "checkpoints": str(checkpoints_path),
                            "metrics": str(metrics_path),
                            "batches": str(batches_path),
                        },
                    },
                },
            )

            self.template.reset_supervised_append_streams(
                log_path,
                include_periodic_eval=False,
                include_checkpoints=False,
                include_train_metrics=False,
                include_batch_trace=False,
            )
            self.template.write_outputs(
                log_path,
                args=self.make_args(),
                eval_examples=[{"id": "eval-1"}],
                predictions=[{"id": "eval-1", "prediction": "ok"}],
                metrics={"eval_accuracy": 1.0},
                include_existing_artifacts=True,
            )

            payload = json.loads((log_path / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(payload["artifacts"]["eval"].keys()),
                ["examples", "metrics", "predictions"],
            )
            self.assertNotIn("train", payload["artifacts"])
            self.assertFalse(periodic_path.exists())
            self.assertFalse(checkpoints_path.exists())
            self.assertFalse(metrics_path.exists())
            self.assertFalse(batches_path.exists())

    def test_eval_only_run_metadata_excludes_stale_training_streams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            periodic_path = log_path / "eval" / "periodic.jsonl"
            checkpoints_path = log_path / "train" / "checkpoints.jsonl"
            metrics_path = log_path / "train" / "metrics.jsonl"
            for path in (periodic_path, checkpoints_path, metrics_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"stale": true}\n', encoding="utf-8")

            args = self.make_args()
            self.template.write_run_metadata(
                log_path,
                args,
                status="running",
                started_at=1.0,
                include_append_streams=False,
            )
            self.template.write_outputs(
                log_path,
                args=args,
                eval_examples=[{"id": "eval-1"}],
                predictions=[{"id": "eval-1", "prediction": "ok"}],
                metrics={"eval_accuracy": 1.0},
                include_existing_artifacts=False,
            )
            self.template.write_run_metadata(
                log_path,
                args,
                status="completed",
                started_at=1.0,
                ended_at=2.0,
                include_append_streams=False,
            )

            payload = json.loads((log_path / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(payload["artifacts"]["eval"].keys()),
                ["examples", "metrics", "predictions"],
            )
            self.assertNotIn("train", payload["artifacts"])
            self.assertNotIn("periodic", payload["artifacts"]["eval"])

    def test_failed_fresh_run_metadata_drops_stale_artifacts_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir)
            train_path = log_path / "train-data.jsonl"
            eval_path = log_path / "eval-data.jsonl"
            train_path.write_text("{}\n", encoding="utf-8")
            eval_path.write_text("{}\n", encoding="utf-8")
            stale_paths = [
                log_path / "eval" / "examples.jsonl",
                log_path / "eval" / "predictions.jsonl",
                log_path / "eval" / "metrics.json",
                log_path / "eval" / "periodic.jsonl",
                log_path / "train" / "checkpoints.jsonl",
                log_path / "train" / "metrics.jsonl",
                log_path / "train" / "batches.jsonl",
            ]
            for path in stale_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"stale": true}\n', encoding="utf-8")

            args = self.make_args(
                log_path=str(log_path),
                train_data=str(train_path),
                eval_data=f"smoke:{eval_path}",
                eval_only=False,
                dry_run=False,
                eval_limit=0,
                tinker_timeout=30,
            )

            with (
                mock.patch.object(self.template, "parse_args", return_value=args),
                mock.patch.object(
                    self.template,
                    "parse_eval_data_arg",
                    return_value=[("smoke", eval_path)],
                ),
                mock.patch.object(
                    self.template,
                    "normalize_eval_rows",
                    return_value=[{"id": "eval-1"}],
                ),
                mock.patch.object(
                    self.template,
                    "normalize_train_rows",
                    return_value=[{"id": "train-1"}],
                ),
                mock.patch.object(
                    self.template,
                    "audit_overlap",
                    return_value={"overlap_count": 0, "overlap_preview": []},
                ),
                mock.patch.object(
                    self.template,
                    "get_last_resumable_checkpoint",
                    return_value=None,
                ),
                mock.patch.object(self.template, "validate_resume_contract"),
                mock.patch.object(
                    self.template,
                    "create_service_client",
                    side_effect=RuntimeError("forced startup failure"),
                ),
                mock.patch.dict(os.environ, {"TINKER_API_KEY": "test-key"}, clear=False),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced startup failure"):
                    asyncio.run(self.template.main_async())

            payload = json.loads((log_path / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertNotIn("artifacts", payload)
            for path in stale_paths:
                self.assertFalse(path.exists(), path)


class ScaffoldTemplateRunTrainTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = load_template_module()

    def make_args(self, **overrides):
        defaults = {
            "base_model": "Qwen/Qwen3-0.6B",
            "seed": 1,
            "rank": 8,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "max_steps": 1,
            "batch_size": 1,
            "lr_schedule": "linear",
            "eval_every": 1,
            "save_every": 0,
            "train_metrics_every": 0,
            "train_print_every": 0,
            "load_checkpoint_path": "",
            "max_concurrent_requests": 1,
            "eval_temperature": 0.0,
            "eval_max_tokens": 32,
            "eval_top_p": 1.0,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    async def run_train_once(self, *, train_metrics_every: int):
        args = self.make_args(train_metrics_every=train_metrics_every)
        fake_client = _FakeTrainingClient()
        train_rows = [{"id": "train-1", "input": "hello", "output": "world"}]
        eval_rows = [{"id": "eval-1", "input": "hello", "output": "world"}]

        async def fake_run_eval(sampler, tokenizer, eval_rows, args):
            return [], [
                {
                    "id": "eval-1",
                    "prediction": "world",
                    "correct": True,
                    "expected": "world",
                    "assistant_text": "world",
                }
            ], {"eval_accuracy": 1.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            patches = (
                mock.patch.object(
                    self.template,
                    "build_supervised_row_datum",
                    side_effect=lambda tokenizer, row: types.SimpleNamespace(
                        loss_fn_inputs={
                            "weights": [1.0],
                            "target_tokens": [1],
                        }
                    ),
                ),
                mock.patch.object(
                    self.template,
                    "build_supervised_batch_trace_record",
                    return_value={"step": 1, "example_ids": ["train-1"]},
                ),
                mock.patch.object(self.template, "run_eval", side_effect=fake_run_eval),
            )
            with patches[0], patches[1], patches[2]:
                await self.template.run_train(
                    fake_client,
                    tokenizer=object(),
                    train_rows=train_rows,
                    eval_rows=eval_rows,
                    args=args,
                    output_dir=output_dir,
                )

            return {
                "periodic_exists": (output_dir / "eval" / "periodic.jsonl").exists(),
                "metrics_exists": (output_dir / "train" / "metrics.jsonl").exists(),
                "batches_exists": (output_dir / "train" / "batches.jsonl").exists(),
                "periodic_text": (
                    (output_dir / "eval" / "periodic.jsonl").read_text(encoding="utf-8")
                    if (output_dir / "eval" / "periodic.jsonl").exists()
                    else ""
                ),
                "metrics_text": (
                    (output_dir / "train" / "metrics.jsonl").read_text(encoding="utf-8")
                    if (output_dir / "train" / "metrics.jsonl").exists()
                    else ""
                ),
                "batches_text": (
                    (output_dir / "train" / "batches.jsonl").read_text(encoding="utf-8")
                    if (output_dir / "train" / "batches.jsonl").exists()
                    else ""
                ),
            }

    async def test_run_train_eval_only_cadence_does_not_create_train_metrics_stream(self):
        paths = await self.run_train_once(train_metrics_every=0)
        self.assertTrue(paths["periodic_exists"])
        self.assertIn('"step": 1', paths["periodic_text"])
        self.assertFalse(paths["metrics_exists"])
        self.assertFalse(paths["batches_exists"])

    async def test_run_train_metrics_cadence_creates_metrics_and_batches(self):
        paths = await self.run_train_once(train_metrics_every=1)
        self.assertTrue(paths["periodic_exists"])
        self.assertTrue(paths["metrics_exists"])
        self.assertTrue(paths["batches_exists"])
        self.assertIn('"test/eval_accuracy": 1.0', paths["metrics_text"])
        self.assertIn('"example_ids": ["train-1"]', paths["batches_text"])


if __name__ == "__main__":
    unittest.main()
