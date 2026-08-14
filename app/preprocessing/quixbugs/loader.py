from __future__ import annotations

from pathlib import Path

from ..dataset_loader import DatasetLoader
from ..raw_benchmark_case import RawBenchmarkCase
from .parser import QuixBugsParser


class QuixBugsLoader(DatasetLoader):
    _EXPECTED_CASES = 40

    def __init__(
        self,
        root: str | Path,
        parser: QuixBugsParser | None = None,
    ):
        self._root = Path(root)
        self._parser = parser or QuixBugsParser()

    def load(
        self,
        *,
        limit: int | None = None,
    ) -> list[RawBenchmarkCase]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        root = self._resolve_dataset_root(self._root)
        test_directory = root / "python_testcases"
        node_path = test_directory / "node.py"
        test_paths = sorted(test_directory.glob("test_*.py"))

        if len(test_paths) != self._EXPECTED_CASES:
            raise ValueError(
                f"Expected {self._EXPECTED_CASES} QuixBugs tests, "
                f"found {len(test_paths)}"
            )

        cases: list[RawBenchmarkCase] = []

        for test_path in test_paths:
            name = test_path.stem.removeprefix("test_")
            case = self._load_case(
                root=root,
                name=name,
                test_path=test_path,
                node_path=node_path,
            )

            cases.append(case)
            if limit is not None and len(cases) >= limit:
                return cases

        return cases

    def _load_case(
        self,
        *,
        root: Path,
        name: str,
        test_path: Path,
        node_path: Path,
    ) -> RawBenchmarkCase:

        buggy_path = root / "python_programs" / f"{name}.py"
        correct_path = root / "correct_python_programs" / f"{name}.py"
        json_path = root / "json_testcases" / f"{name}.json"

        if not buggy_path.is_file():
            raise FileNotFoundError(f"Missing QuixBugs buggy program: {buggy_path}")

        if not correct_path.is_file():
            raise FileNotFoundError(f"Missing QuixBugs correct program: {correct_path}")

        test_text = test_path.read_text(encoding="utf-8")

        if json_path.is_file():
            tests = self._parser.parse_json_tests(
                name=name,
                text=test_text,
                json_text=json_path.read_text(encoding="utf-8"),
                source=test_path,
            )
        else:
            node_text = ""
            if node_path.is_file():
                node_text = node_path.read_text(encoding="utf-8")

            tests = self._parser.parse_python_tests(
                name=name,
                text=test_text,
                node_text=node_text,
                source=test_path,
                node_source=node_path,
            )

        return self._parser.parse_program(
            name=name,
            buggy_text=buggy_path.read_text(encoding="utf-8"),
            correct_text=correct_path.read_text(encoding="utf-8"),
            tests=tests,
            buggy_path=buggy_path,
            correct_path=correct_path,
        )

    @staticmethod
    def _resolve_dataset_root(
        root: Path,
    ) -> Path:
        if (root / "python_programs").is_dir():
            return root

        matches = []

        for path in root.rglob("python_programs"):
            candidate = path.parent

            if (candidate / "correct_python_programs").is_dir() and (
                candidate / "python_testcases"
            ).is_dir():
                matches.append(candidate)

        if len(matches) != 1:
            raise ValueError(f"Could not uniquely locate QuixBugs dataset under {root}")

        return matches[0]
