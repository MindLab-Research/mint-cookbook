import json
import os
import select
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = EXPERIMENT_DIR / ".env"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CACHE_ROOT = str(Path.home() / ".cache" / "mint-cookbook")
DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")


def ensure_writable_dir(path: Path) -> Path | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    probe = path / ".write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return None
    return path


def choose_cache_dir(preferred: str, fallback_name: str) -> Path:
    preferred_path = Path(preferred).expanduser()
    writable = ensure_writable_dir(preferred_path)
    if writable is not None:
        return writable
    fallback = Path(tempfile.gettempdir()) / "mint-cookbook" / fallback_name
    writable = ensure_writable_dir(fallback)
    if writable is None:
        raise RuntimeError(f"unable to create writable cache dir: {preferred_path} or {fallback}")
    return writable


def load_live_env() -> dict[str, str]:
    env = dict(os.environ)
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
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

    cache_root = choose_cache_dir(
        env.get("MINT_SHARED_CACHE_ROOT", DEFAULT_CACHE_ROOT),
        "dapo-aime-cache",
    )
    env["MINT_SHARED_CACHE_ROOT"] = str(cache_root)
    env["UV_CACHE_DIR"] = str(
        choose_cache_dir(env.get("UV_CACHE_DIR", str(cache_root / "uv")), "dapo-aime-uv")
    )
    env["XDG_CACHE_HOME"] = str(
        choose_cache_dir(env.get("XDG_CACHE_HOME", str(cache_root / "xdg")), "dapo-aime-xdg")
    )
    env["PIP_CACHE_DIR"] = str(
        choose_cache_dir(env.get("PIP_CACHE_DIR", str(cache_root / "pip")), "dapo-aime-pip")
    )
    env["MPLCONFIGDIR"] = str(
        choose_cache_dir(env.get("MPLCONFIGDIR", str(cache_root / "matplotlib")), "dapo-aime-matplotlib")
    )
    env["WANDB_DIR"] = str(
        choose_cache_dir(env.get("WANDB_DIR", str(cache_root / "wandb")), "dapo-aime-wandb")
    )
    hf_home = choose_cache_dir(env.get("HF_HOME", DEFAULT_HF_HOME), "dapo-aime-hf")
    env["HF_HOME"] = str(hf_home)
    hub_cache = choose_cache_dir(
        env.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")),
        "dapo-aime-hf-hub",
    )
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    env["TRANSFORMERS_CACHE"] = str(
        choose_cache_dir(env.get("TRANSFORMERS_CACHE", str(hub_cache)), "dapo-aime-transformers")
    )
    env["HF_DATASETS_CACHE"] = str(
        choose_cache_dir(
            env.get("HF_DATASETS_CACHE", str(cache_root / "hf-datasets")),
            "dapo-aime-datasets",
        )
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def run_completed(
    cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_s: int
) -> subprocess.CompletedProcess[str]:
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
        if row_step == step and row.get("state_path") and row.get("sampler_path"):
            return True
    return False


def latest_state_path(path: Path) -> str:
    for row in reversed(read_jsonl(path)):
        state_path = str(row.get("state_path") or "").strip()
        if state_path:
            return state_path
    raise AssertionError(f"missing resumable state_path in {path}")


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

        remaining = (
            proc.stdout.read()
            if proc.stdout is not None and proc.poll() is not None
            else ""
        )
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


def make_train_rows() -> list[dict]:
    return [
        {
            "id": "train-1",
            "question": "What is 1+1?",
            "answer": "2",
            "source": "tests",
        }
    ]


def make_eval_rows() -> list[dict]:
    return [
        {
            "ID": "2024-I-1",
            "Problem": "What is 1+1?",
            "Answer": 2,
        }
    ]


class LiveDAPOAIMEFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = load_live_env()
        if not cls.env.get("MINT_API_KEY"):
            raise RuntimeError("missing live MINT_API_KEY for dapo-aime tests")

    def test_eval_only_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="dapo-aime-eval-only-") as tmpdir:
            root = Path(tmpdir)
            eval_path = root / "eval.jsonl"
            log_path = root / "eval-only"
            write_jsonl(eval_path, make_eval_rows())

            completed = run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--eval-only",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--eval-data",
                    str(eval_path),
                    "--eval-num-samples",
                    "1",
                    "--eval-temperature",
                    "0",
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    "1",
                    "--mint-timeout",
                    "1200",
                    "--log-path",
                    str(log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=2400,
            )
            run_data = read_json(log_path / "run.json")
            metrics = read_json(log_path / "eval" / "metrics.json")
            predictions = read_jsonl(log_path / "eval" / "predictions.jsonl")
            self.assertEqual(run_data["status"], "completed")
            self.assertTrue(run_data["args"]["eval_only"])
            self.assertEqual(run_data["args"]["eval_data"], str(eval_path))
            self.assertNotIn("train", run_data.get("artifacts", {}))
            self.assertEqual(len(predictions), 1)
            self.assertIn("eval_accuracy", metrics)
            self.assertEqual(run_data["eval_metrics"]["eval_accuracy"], metrics["eval_accuracy"])
            self.assertIn("METRIC eval_accuracy=", completed.stdout)

    def test_train_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="dapo-aime-train-") as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            log_path = root / "train"
            write_jsonl(train_path, make_train_rows())
            write_jsonl(eval_path, make_eval_rows())

            completed = run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--train-data",
                    str(train_path),
                    "--eval-data",
                    str(eval_path),
                    "--rank",
                    "4",
                    "--grpo-steps",
                    "1",
                    "--groups-per-batch",
                    "1",
                    "--group-size",
                    "1",
                    "--stream-minibatches-per-step",
                    "1",
                    "--save-every-steps",
                    "1",
                    "--eval-every-steps",
                    "0",
                    "--dynamic-sampling-type",
                    "none",
                    "--rl-max-tokens",
                    "64",
                    "--eval-num-samples",
                    "1",
                    "--eval-temperature",
                    "0",
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    "1",
                    "--tail-grace-seconds",
                    "1",
                    "--mint-timeout",
                    "1200",
                    "--client-create-timeout",
                    "600",
                    "--log-path",
                    str(log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=3600,
            )
            run_data = read_json(log_path / "run.json")
            train_metrics = read_jsonl(log_path / "train" / "metrics.jsonl")
            checkpoint_rows = read_jsonl(log_path / "train" / "checkpoints.jsonl")
            eval_metrics = read_json(log_path / "eval" / "metrics.json")
            predictions = read_jsonl(log_path / "eval" / "predictions.jsonl")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["args"]["train_data"], str(train_path))
            self.assertEqual(run_data["args"]["eval_data"], str(eval_path))
            self.assertEqual(run_data["args"]["resume_completed_steps"], 0)
            self.assertEqual([row["step"] for row in train_metrics], [1])
            self.assertEqual(checkpoint_rows[0]["step"], 1)
            self.assertEqual(checkpoint_rows[0]["completed_steps"], 1)
            self.assertIn("state_path", checkpoint_rows[0])
            self.assertIn("sampler_path", checkpoint_rows[0])
            self.assertEqual(len(predictions), 1)
            self.assertIn("eval_accuracy", eval_metrics)
            self.assertIn("METRIC rl_reward_mean=", completed.stdout)
            self.assertIn("METRIC eval_accuracy=", completed.stdout)
            self.assertIn("@@ checkpoint step=1/1", completed.stdout)

    def test_resume_live_smoke(self):
        with tempfile.TemporaryDirectory(prefix="dapo-aime-resume-") as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            log_path = root / "resume"
            checkpoints_path = log_path / "train" / "checkpoints.jsonl"
            write_jsonl(train_path, make_train_rows())
            write_jsonl(eval_path, make_eval_rows())

            base_cmd = [
                "uv",
                "run",
                "train.py",
                "--base-model",
                DEFAULT_BASE_MODEL,
                "--train-data",
                str(train_path),
                "--eval-data",
                str(eval_path),
                "--rank",
                "4",
                "--grpo-steps",
                "2",
                "--groups-per-batch",
                "1",
                "--group-size",
                "1",
                "--stream-minibatches-per-step",
                "1",
                "--save-every-steps",
                "1",
                "--eval-every-steps",
                "0",
                "--dynamic-sampling-type",
                "none",
                "--rl-max-tokens",
                "64",
                "--eval-num-samples",
                "1",
                "--eval-temperature",
                "0",
                "--eval-max-tokens",
                "64",
                "--max-concurrent-requests",
                "1",
                "--tail-grace-seconds",
                "1",
                "--mint-timeout",
                "1200",
                "--client-create-timeout",
                "600",
                "--log-path",
                str(log_path),
            ]
            interrupted_output = interrupt_after_checkpoint(
                base_cmd,
                cwd=EXPERIMENT_DIR,
                env=self.env,
                checkpoints_path=checkpoints_path,
                step=1,
                timeout_s=3600,
            )
            self.assertIn("@@ checkpoint step=1/2", interrupted_output)
            checkpoint_rows = read_jsonl(checkpoints_path)
            self.assertEqual(checkpoint_rows[0]["step"], 1)
            resume_from = latest_state_path(checkpoints_path)

            resumed = run_completed(
                base_cmd,
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=4800,
            )
            self.assertIn(f"Resuming from saved state: {resume_from}", resumed.stdout)
            self.assertIn(
                ">> train_resume completed_steps=1 next_step=2",
                resumed.stdout,
            )
            run_data = read_json(log_path / "run.json")
            train_metrics = read_jsonl(log_path / "train" / "metrics.jsonl")
            checkpoint_rows = read_jsonl(checkpoints_path)
            eval_metrics = read_json(log_path / "eval" / "metrics.json")
            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["args"]["resume_completed_steps"], 1)
            self.assertEqual([row["step"] for row in train_metrics], [1, 2])
            self.assertEqual([row["step"] for row in checkpoint_rows], [1, 2])
            self.assertTrue(all(row.get("state_path") for row in checkpoint_rows))
            self.assertTrue(all(row.get("sampler_path") for row in checkpoint_rows))
            self.assertIn("eval_accuracy", eval_metrics)

    def test_eval_only_from_saved_sampler_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="dapo-aime-checkpoint-eval-") as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            train_log_path = root / "train"
            eval_log_path = root / "eval-from-checkpoint"
            write_jsonl(train_path, make_train_rows())
            write_jsonl(eval_path, make_eval_rows())

            run_completed(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--base-model",
                    DEFAULT_BASE_MODEL,
                    "--train-data",
                    str(train_path),
                    "--eval-data",
                    str(eval_path),
                    "--rank",
                    "4",
                    "--grpo-steps",
                    "1",
                    "--groups-per-batch",
                    "1",
                    "--group-size",
                    "1",
                    "--stream-minibatches-per-step",
                    "1",
                    "--save-every-steps",
                    "1",
                    "--eval-every-steps",
                    "0",
                    "--dynamic-sampling-type",
                    "none",
                    "--rl-max-tokens",
                    "64",
                    "--eval-num-samples",
                    "1",
                    "--eval-temperature",
                    "0",
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    "1",
                    "--tail-grace-seconds",
                    "1",
                    "--mint-timeout",
                    "1200",
                    "--client-create-timeout",
                    "600",
                    "--log-path",
                    str(train_log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=3600,
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
                    str(eval_path),
                    "--eval-num-samples",
                    "1",
                    "--eval-temperature",
                    "0",
                    "--eval-max-tokens",
                    "64",
                    "--max-concurrent-requests",
                    "1",
                    "--mint-timeout",
                    "1200",
                    "--log-path",
                    str(eval_log_path),
                ],
                cwd=EXPERIMENT_DIR,
                env=self.env,
                timeout_s=2400,
            )
            run_data = read_json(eval_log_path / "run.json")
            metrics = read_json(eval_log_path / "eval" / "metrics.json")
            predictions = read_jsonl(eval_log_path / "eval" / "predictions.jsonl")
            self.assertEqual(run_data["status"], "completed")
            self.assertTrue(run_data["args"]["eval_only"])
            self.assertEqual(run_data["args"]["base_model"], sampler_path)
            self.assertEqual(run_data["args"].get("load_checkpoint_path", ""), "")
            self.assertEqual(len(predictions), 1)
            self.assertIn("eval_accuracy", metrics)
            self.assertIn("METRIC eval_accuracy=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
