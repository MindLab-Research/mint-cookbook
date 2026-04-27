import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
TEST_ENV_PATH = EXPERIMENT_DIR / "tests" / ".env"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")
TRAIN_SMOKE_PATH = EXPERIMENT_DIR / "data" / "train" / "smoke.jsonl"
EVAL_SMOKE_PATH = EXPERIMENT_DIR / "data" / "eval" / "smoke.jsonl"
TRAIN_EVAL_SMOKE_PATH = EXPERIMENT_DIR / "data" / "eval" / "smoke.jsonl"
EVAL_SMOKE_TASK_IDS = ("1-2", "2-5", "2-7", "3-3", "3-4")
SMOKE_MAX_CONCURRENT_REQUESTS = 8


def load_live_env() -> dict[str, str]:
    env = dict(os.environ)
    if TEST_ENV_PATH.exists():
        for line in TEST_ENV_PATH.read_text(encoding="utf-8").splitlines():
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


def latest_sampler_path(path: Path) -> str:
    for row in reversed(read_jsonl(path)):
        sampler_path = str(row.get("sampler_path") or "").strip()
        if sampler_path:
            return sampler_path
    raise AssertionError(f"missing sampler_path in {path}")


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


def interrupt_after_step_one(
    cmd: list[str], *, cwd: Path, env: dict[str, str], wait_pattern: str, timeout_s: int
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
            assert proc.stdout is not None
            line = proc.stdout.readline()
            if line:
                lines.append(line)
                if wait_pattern in line:
                    break
                continue
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
            raise AssertionError(f"timed out waiting for `{wait_pattern}`\noutput tail:\n{tail(''.join(lines))}")

        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
        remaining = proc.stdout.read() if proc.stdout is not None else ""
        output = "".join(lines) + (remaining or "")
        if wait_pattern not in output:
            raise AssertionError(f"missing `{wait_pattern}` in interrupted output\noutput tail:\n{tail(output)}")
        return output
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


class LiveLawBenchFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = load_live_env()
        missing = [key for key in ("MINT_BASE_URL", "MINT_API_KEY") if not cls.env.get(key)]
        if missing:
            raise RuntimeError(f"missing live MinT env for LawBench tests: {', '.join(missing)}")
        for path in (TRAIN_SMOKE_PATH, EVAL_SMOKE_PATH, TRAIN_EVAL_SMOKE_PATH):
            if not path.is_file():
                raise RuntimeError(f"missing smoke data path: {path}")
        eval_smoke_rows = read_jsonl(EVAL_SMOKE_PATH)
        eval_smoke_task_ids = tuple(row["task_id"] for row in eval_smoke_rows)
        if eval_smoke_task_ids != EVAL_SMOKE_TASK_IDS:
            raise RuntimeError(
                "unexpected eval smoke task ids: "
                f"expected {EVAL_SMOKE_TASK_IDS}, got {eval_smoke_task_ids}"
            )
        if len(set(eval_smoke_task_ids)) != len(eval_smoke_task_ids):
            raise RuntimeError("eval smoke rows must keep exactly one row per smoke task")
        cls.eval_smoke_row_count = len(eval_smoke_rows)
        cls.eval_smoke_task_count = len(eval_smoke_task_ids)

    def test_eval_only_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="lawbench-eval-only-") as tmpdir:
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
                    str(EVAL_SMOKE_PATH),
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    str(SMOKE_MAX_CONCURRENT_REQUESTS),
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
            self.assertEqual(run_data["eval_data"], str(EVAL_SMOKE_PATH))
            self.assertNotIn("train", run_data.get("artifacts", {}))
            self.assertIn("eval_lawbench_avg", metrics)
            self.assertEqual(len(predictions), self.eval_smoke_row_count)
            self.assertEqual([row["task_id"] for row in predictions], list(EVAL_SMOKE_TASK_IDS))
            self.assertEqual(len(metrics.get("eval_task_metrics", {})), self.eval_smoke_task_count)
            self.assertEqual(sorted(metrics.get("eval_task_metrics", {})), list(EVAL_SMOKE_TASK_IDS))

    def test_train_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="lawbench-train-") as tmpdir:
            log_path = Path(tmpdir) / "train"
            run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--train-data",
                    str(TRAIN_SMOKE_PATH),
                    "--eval-data",
                    str(EVAL_SMOKE_PATH),
                    "--train-eval-data",
                    str(TRAIN_EVAL_SMOKE_PATH),
                    "--batch-size",
                    "2",
                    "--rank",
                    "4",
                    "--num-epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--save-every",
                    "1",
                    "--eval-every",
                    "0",
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    str(SMOKE_MAX_CONCURRENT_REQUESTS),
                    "--log-path",
                    str(log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=2400,
            )
            run_data = read_json(log_path / "run.json")
            train_metrics = read_jsonl(log_path / "train" / "metrics.jsonl")
            checkpoint_rows = read_jsonl(log_path / "train" / "checkpoints.jsonl")
            eval_metrics = read_json(log_path / "eval" / "metrics.json")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["train_data"], str(TRAIN_SMOKE_PATH))
            self.assertEqual(run_data["eval_data"], str(EVAL_SMOKE_PATH))
            self.assertEqual(run_data["train_eval_data"], str(TRAIN_EVAL_SMOKE_PATH))
            self.assertEqual([row["step"] for row in train_metrics], [1])
            self.assertEqual(checkpoint_rows[0]["step"], 1)
            self.assertTrue(checkpoint_rows[-1]["final"])
            self.assertIn("state_path", checkpoint_rows[-1])
            self.assertIn("sampler_path", checkpoint_rows[-1])
            self.assertIn("eval_lawbench_avg", eval_metrics)
            self.assertEqual(len(eval_metrics.get("eval_task_metrics", {})), self.eval_smoke_task_count)
            self.assertEqual(sorted(eval_metrics.get("eval_task_metrics", {})), list(EVAL_SMOKE_TASK_IDS))

    def test_resume_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="lawbench-resume-") as tmpdir:
            log_path = Path(tmpdir) / "resume"
            base_cmd = [
                "uv",
                "run",
                "train.py",
                "--base-model",
                DEFAULT_BASE_MODEL,
                "--train-data",
                str(TRAIN_SMOKE_PATH),
                "--eval-data",
                str(EVAL_SMOKE_PATH),
                "--train-eval-data",
                str(TRAIN_EVAL_SMOKE_PATH),
                "--batch-size",
                "2",
                "--rank",
                "4",
                "--num-epochs",
                "1",
                "--max-steps",
                "2",
                "--save-every",
                "1",
                "--eval-every",
                "0",
                "--eval-max-tokens",
                "64",
                "--max-concurrent-requests",
                str(SMOKE_MAX_CONCURRENT_REQUESTS),
                "--log-path",
                str(log_path),
            ]
            interrupted_output = interrupt_after_step_one(
                base_cmd,
                cwd=EXPERIMENT_DIR,
                env=self.env,
                wait_pattern="Checkpoint at step 1:",
                timeout_s=1800,
            )
            self.assertIn("Checkpoint at step 1:", interrupted_output)
            checkpoint_rows = read_jsonl(log_path / "train" / "checkpoints.jsonl")
            self.assertEqual(checkpoint_rows[0]["step"], 1)

            resumed = run_completed(
                base_cmd,
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=3000,
            )
            self.assertIn("Resume bookkeeping: start_step=1", resumed.stdout)
            self.assertIn("Resuming from saved state:", resumed.stdout)
            run_data = read_json(log_path / "run.json")
            train_metrics = read_jsonl(log_path / "train" / "metrics.jsonl")
            checkpoint_rows = read_jsonl(log_path / "train" / "checkpoints.jsonl")
            eval_metrics = read_json(log_path / "eval" / "metrics.json")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual([row["step"] for row in train_metrics], [1, 2])
            self.assertEqual(checkpoint_rows[0]["step"], 1)
            self.assertEqual(checkpoint_rows[1]["step"], 2)
            self.assertTrue(checkpoint_rows[-1]["final"])
            self.assertIn("eval_lawbench_avg", eval_metrics)
            self.assertEqual(len(eval_metrics.get("eval_task_metrics", {})), self.eval_smoke_task_count)
            self.assertEqual(sorted(eval_metrics.get("eval_task_metrics", {})), list(EVAL_SMOKE_TASK_IDS))

    def test_eval_only_from_saved_sampler_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="lawbench-checkpoint-eval-") as tmpdir:
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
                    str(TRAIN_SMOKE_PATH),
                    "--eval-data",
                    str(EVAL_SMOKE_PATH),
                    "--train-eval-data",
                    str(TRAIN_EVAL_SMOKE_PATH),
                    "--batch-size",
                    "2",
                    "--rank",
                    "4",
                    "--num-epochs",
                    "1",
                    "--max-steps",
                    "1",
                    "--save-every",
                    "1",
                    "--eval-every",
                    "0",
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    str(SMOKE_MAX_CONCURRENT_REQUESTS),
                    "--log-path",
                    str(train_log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=2400,
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
                    str(EVAL_SMOKE_PATH),
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    str(SMOKE_MAX_CONCURRENT_REQUESTS),
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
            self.assertEqual(run_data["args"]["base_model"], sampler_path)
            self.assertEqual(run_data["args"].get("load_checkpoint_path", ""), "")
            self.assertEqual(len(predictions), self.eval_smoke_row_count)
            self.assertIn("eval_lawbench_avg", metrics)
            self.assertEqual(sorted(metrics.get("eval_task_metrics", {})), list(EVAL_SMOKE_TASK_IDS))
            self.assertIn("METRIC eval_lawbench_avg=", completed.stdout)
if __name__ == "__main__":
    unittest.main()
