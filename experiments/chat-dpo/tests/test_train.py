import json
import os
import re
import select
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ENV_PATHS = (
    EXPERIMENT_DIR / "tests" / ".env",
    EXPERIMENT_DIR / ".env",
)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_SLUG = re.sub(r"[^a-z0-9]+", "-", DEFAULT_BASE_MODEL.lower()).strip("-")
DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")
TRAIN_PATH = EXPERIMENT_DIR / "data" / "train.jsonl"
EVAL_PATH = EXPERIMENT_DIR / "data" / "eval.jsonl"


def load_live_env() -> dict[str, str]:
    env = dict(os.environ)
    for env_path in ENV_PATHS:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].lstrip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key not in {"MINT_BASE_URL", "MINT_API_KEY"}:
                continue
            env.setdefault(key, value.strip().strip('"').strip("'"))
    env.setdefault("HF_HOME", DEFAULT_HF_HOME)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def run_completed(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed with exit code {completed.returncode}\n"
            f"stdout/stderr tail:\n{tail((completed.stdout or '') + (completed.stderr or ''))}"
        )
    return completed


def checkpoint_step_recorded(path: Path, step: int) -> bool:
    if not path.is_file():
        return False
    for row in read_jsonl(path):
        try:
            row_step = int(row.get("step"))
        except (TypeError, ValueError):
            continue
        if row_step == step and row.get("state_path"):
            return True
    return False


def latest_sampler_path(path: Path) -> str:
    for row in reversed(read_jsonl(path)):
        sampler_path = str(row.get("sampler_path") or "").strip()
        if sampler_path:
            return sampler_path
    raise AssertionError(f"missing sampler_path in {path}")


def interrupt_after_checkpoint(
    cmd: list[str], *, cwd: Path, env: dict[str, str], checkpoints_path: Path, step: int, timeout_s: int
) -> str:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    lines: list[str] = []
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            if proc.stdout is not None:
                ready, _, _ = select.select([proc.stdout], [], [], 0.2)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        lines.append(line)
            if checkpoint_step_recorded(checkpoints_path, step):
                break
            if proc.poll() is not None:
                break
        else:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
            raise AssertionError(
                f"timed out waiting for checkpoint step {step}\noutput tail:\n{tail(''.join(lines))}"
            )

        remaining = proc.stdout.read() if proc.stdout is not None and proc.poll() is not None else ""
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            remaining = proc.stdout.read() if proc.stdout is not None else ""
        output = "".join(lines) + (remaining or "")
        if not checkpoint_step_recorded(checkpoints_path, step):
            raise AssertionError(
                f"missing checkpoint step {step} after interrupt\noutput tail:\n{tail(output)}"
            )
        return output
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


class LiveChatDPOFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = load_live_env()
        missing = [key for key in ("MINT_BASE_URL", "MINT_API_KEY") if not cls.env.get(key)]
        if missing:
            raise RuntimeError(f"missing live MinT env for chat-dpo tests: {', '.join(missing)}")
        for path in (TRAIN_PATH, EVAL_PATH):
            if not path.is_file():
                raise RuntimeError(f"missing live test data path: {path}")

    def test_eval_only_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="chat-dpo-eval-only-") as tmpdir:
            log_path = Path(tmpdir) / "eval-only"
            run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--eval-only",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--eval-data",
                    "data/eval.jsonl",
                    "--eval-limit",
                    "2",
                    "--max-concurrent-requests",
                    "1",
                    "--log-path",
                    str(log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=1800,
            )
            run_data = read_json(log_path / "run.json")
            metrics = read_json(log_path / "eval" / "metrics.json")
            predictions = read_jsonl(log_path / "eval" / "predictions.jsonl")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["args"]["eval_data"], "data/eval.jsonl")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(metrics["eval_num_pairs"], 2.0)
            self.assertIn("eval_pair_accuracy", metrics)

    def test_train_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="chat-dpo-train-") as tmpdir:
            log_path = Path(tmpdir) / "train"
            completed = run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--train-data",
                    "data/train.jsonl",
                    "--eval-data",
                    "data/eval.jsonl",
                    "--batch-size",
                    "2",
                    "--num-epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--save-every",
                    "1",
                    "--eval-every",
                    "0",
                    "--max-concurrent-requests",
                    "1",
                    "--log-path",
                    str(log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=3000,
            )
            run_data = read_json(log_path / "run.json")
            train_metrics = read_jsonl(log_path / "train" / "metrics.jsonl")
            checkpoint_rows = read_jsonl(log_path / "train" / "checkpoints.jsonl")
            eval_metrics = read_json(log_path / "eval" / "metrics.json")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["reference_model"], DEFAULT_BASE_MODEL)
            self.assertNotIn("run_id", run_data)
            self.assertIn("METRIC train_steps=1.0000", completed.stdout)
            self.assertIn("METRIC train_total_steps=1.0000", completed.stdout)
            self.assertIn("METRIC train_duration_seconds=", completed.stdout)
            self.assertNotIn("train_steps_completed", completed.stdout)
            self.assertEqual([row["step"] for row in train_metrics], [1])
            self.assertEqual(checkpoint_rows[0]["step"], 1)
            self.assertEqual(checkpoint_rows[0]["epoch"], 1)
            self.assertEqual(checkpoint_rows[0]["batch"], 0)
            self.assertEqual(
                checkpoint_rows[0]["name"],
                f"chat-dpo-train-{DEFAULT_MODEL_SLUG}-dpo-step-000001",
            )
            self.assertEqual(checkpoint_rows[-1]["step"], 1)
            self.assertEqual(checkpoint_rows[-1]["epoch"], 1)
            self.assertEqual(checkpoint_rows[-1]["batch"], 0)
            self.assertTrue(checkpoint_rows[-1]["final"])
            self.assertEqual(
                checkpoint_rows[-1]["name"],
                f"chat-dpo-train-{DEFAULT_MODEL_SLUG}-dpo",
            )
            self.assertIn("state_path", checkpoint_rows[-1])
            self.assertIn("sampler_path", checkpoint_rows[-1])
            self.assertEqual(run_data["state_path"], checkpoint_rows[-1]["state_path"])
            self.assertEqual(run_data["sampler_path"], checkpoint_rows[-1]["sampler_path"])
            self.assertEqual(eval_metrics["eval_num_pairs"], 2.0)

    def test_resume_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="chat-dpo-resume-") as tmpdir:
            log_path = Path(tmpdir) / "resume"
            checkpoints_path = log_path / "train" / "checkpoints.jsonl"
            base_cmd = [
                "uv",
                "run",
                "train.py",
                "--base-model",
                DEFAULT_BASE_MODEL,
                "--train-data",
                "data/train.jsonl",
                "--eval-data",
                "data/eval.jsonl",
                "--batch-size",
                "1",
                "--num-epochs",
                "1",
                "--max-steps",
                "2",
                "--save-every",
                "1",
                "--eval-every",
                "0",
                "--train-metrics-every",
                "1",
                "--train-print-every",
                "1",
                "--max-concurrent-requests",
                "1",
                "--log-path",
                str(log_path),
            ]
            interrupt_after_checkpoint(
                base_cmd,
                cwd=EXPERIMENT_DIR,
                env=self.env,
                checkpoints_path=checkpoints_path,
                step=1,
                timeout_s=3000,
            )
            checkpoint_rows = read_jsonl(checkpoints_path)
            self.assertEqual(checkpoint_rows[0]["step"], 1)

            resumed = run_completed(
                base_cmd,
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=3600,
            )
            self.assertIn("Resuming from saved state:", resumed.stdout)
            self.assertIn("Resume bookkeeping: start_step=1", resumed.stdout)
            run_data = read_json(log_path / "run.json")
            train_metrics = read_jsonl(log_path / "train" / "metrics.jsonl")
            checkpoint_rows = read_jsonl(checkpoints_path)
            eval_metrics = read_json(log_path / "eval" / "metrics.json")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual([row["step"] for row in train_metrics], [1, 2])
            self.assertEqual(checkpoint_rows[0]["step"], 1)
            self.assertEqual(checkpoint_rows[0]["epoch"], 0)
            self.assertEqual(checkpoint_rows[0]["batch"], 1)
            self.assertEqual(checkpoint_rows[1]["step"], 2)
            self.assertEqual(checkpoint_rows[1]["epoch"], 1)
            self.assertEqual(checkpoint_rows[1]["batch"], 0)
            self.assertTrue(checkpoint_rows[-1]["final"])
            self.assertEqual(eval_metrics["eval_num_pairs"], 2.0)

    def test_eval_only_from_saved_sampler_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="chat-dpo-checkpoint-eval-") as tmpdir:
            root = Path(tmpdir)
            train_log_path = root / "train"
            eval_log_path = root / "eval-from-checkpoint"
            run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--train-data",
                    "data/train.jsonl",
                    "--eval-data",
                    "data/eval.jsonl",
                    "--batch-size",
                    "2",
                    "--num-epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--save-every",
                    "1",
                    "--eval-every",
                    "0",
                    "--max-concurrent-requests",
                    "1",
                    "--log-path",
                    str(train_log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=3000,
            )
            sampler_path = latest_sampler_path(train_log_path / "train" / "checkpoints.jsonl")

            completed = run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--eval-only",
                    "--base-model",
                    sampler_path,
                    "--eval-data",
                    "data/eval.jsonl",
                    "--eval-limit",
                    "2",
                    "--max-concurrent-requests",
                    "1",
                    "--log-path",
                    str(eval_log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=1800,
            )
            run_data = read_json(eval_log_path / "run.json")
            metrics = read_json(eval_log_path / "eval" / "metrics.json")
            predictions = read_jsonl(eval_log_path / "eval" / "predictions.jsonl")
            self.assertEqual(run_data["status"], "completed")
            self.assertTrue(run_data["args"]["eval_only"])
            self.assertEqual(run_data["args"]["base_model"], sampler_path)
            self.assertEqual(run_data["args"].get("load_checkpoint_path", ""), "")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(metrics["eval_num_pairs"], 2.0)
            self.assertIn("METRIC eval_pair_accuracy=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
