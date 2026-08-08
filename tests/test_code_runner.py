import subprocess
from pathlib import Path

import pytest
from app.execution.code_runner import (
    DockerCodeRunner,
)


def test_runner_writes_files_and_returns_result(
    monkeypatch,
):
    runner = DockerCodeRunner(
        image="test-image",
    )

    monkeypatch.setattr(
        DockerCodeRunner,
        "is_available",
        staticmethod(lambda: True),
    )

    def fake_run(
        command,
        *,
        capture_output,
        text,
        timeout,
        check,
    ):
        assert command[:2] == ["docker", "run"]
        assert "--network" in command
        assert "none" in command
        assert "test-image" in command

        mount_index = command.index("--mount") + 1
        mount = command[mount_index]

        prefix = "type=bind,source="
        suffix = ",target=/workspace,readonly"

        assert mount.startswith(prefix)
        assert mount.endswith(suffix)

        workspace = Path(mount[len(prefix) : -len(suffix)])

        assert (workspace / "solution.py").read_text(encoding="utf-8") == (
            "def add(a, b):\n    return a + b\n"
        )

        assert (workspace / "test_solution.py").read_text(encoding="utf-8") == (
            "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        )

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    result = runner.run(
        code=("def add(a, b):\n    return a + b\n"),
        tests=(
            "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        ),
    )

    assert result.passed
    assert result.exit_code == 0
    assert result.output == "1 passed"


def test_runner_rejects_empty_input():
    runner = DockerCodeRunner()

    with pytest.raises(ValueError):
        runner.run(
            code="",
            tests="def test_example(): pass",
        )

    with pytest.raises(ValueError):
        runner.run(
            code="pass",
            tests="",
        )
