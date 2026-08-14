from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TestRunResult:
    __test__ = False

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        parts = []
        for part in (self.stdout.strip(), self.stderr.strip()):
            if part:
                parts.append(part)

        output = "\n".join(parts)

        if len(output) > 4000:
            output = "[output truncated]\n" + output[-4000:]

        return output


class CodeRunner(Protocol):
    def run(
        self,
        *,
        code: str,
        tests: str,
    ) -> TestRunResult: ...


class DockerCodeRunner:
    """Runs Python code and pytest tests inside a restricted container."""

    def __init__(
        self,
        image: str = "multi-debug-sandbox:latest",
        timeout_seconds: int = 15,
    ):
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")

        self.image = image
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def is_available() -> bool:
        return shutil.which("docker") is not None

    def run(
        self,
        *,
        code: str,
        tests: str,
    ) -> TestRunResult:
        if not code.strip():
            raise ValueError("code must not be empty")

        if not tests.strip():
            raise ValueError("tests must not be empty")

        if not self.is_available():
            raise RuntimeError("Docker is unavailable. Start Docker Desktop first.")

        container_name = f"multi-debug-{uuid.uuid4().hex}"

        with tempfile.TemporaryDirectory(
            prefix="multi-debug-",
        ) as temporary_directory:
            workspace = Path(temporary_directory)

            (workspace / "solution.py").write_text(
                code,
                encoding="utf-8",
            )

            (workspace / "test_solution.py").write_text(
                tests,
                encoding="utf-8",
            )

            command = self._build_command(
                workspace=workspace,
                container_name=container_name,
            )

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
                if completed.returncode in (125, 126, 127):
                    message = completed.stderr.strip()
                    if not message:
                        message = completed.stdout.strip()

                    raise RuntimeError(
                        f"Docker execution failed with exit code "
                        f"{completed.returncode}: {message}"
                    )
            except subprocess.TimeoutExpired as error:
                self._remove_container(container_name)

                return TestRunResult(
                    passed=False,
                    exit_code=124,
                    stdout=self._as_text(error.stdout),
                    stderr=(self._as_text(error.stderr) or "Test execution timed out."),
                    timed_out=True,
                )

        return TestRunResult(
            passed=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=(completed.stdout or "")[-10_000:],
            stderr=(completed.stderr or "")[-10_000:],
        )

    def _build_command(
        self,
        *,
        workspace: Path,
        container_name: str,
    ) -> list[str]:
        mount = f"type=bind,source={workspace.resolve()},target=/workspace,readonly"

        return [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--pids-limit",
            "64",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            mount,
            self.image,
            "python",
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            "--maxfail=2",
            "/workspace/test_solution.py",
        ]

    @staticmethod
    def _remove_container(
        container_name: str,
    ) -> None:
        subprocess.run(
            [
                "docker",
                "rm",
                "-f",
                container_name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )

    @staticmethod
    def _as_text(
        value: str | bytes | None,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, bytes):
            return value.decode(
                "utf-8",
                errors="replace",
            )

        return value
