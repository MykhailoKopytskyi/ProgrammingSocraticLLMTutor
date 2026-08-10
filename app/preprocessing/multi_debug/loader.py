from __future__ import annotations

from pathlib import Path

from ..dataset_loader import DatasetLoader
from ..raw_benchmark_case import RawBenchmarkCase
from .parser import MultiDebugParser


class MultiDebugLoader(DatasetLoader):
    _VARIANTS = (
        ("1bug-MULTI_BUG", 1),
        ("2bug-MULTI-BUG", 2),
        ("3bug-MULTI-BUG", 3),
    )

    def __init__(self, root: str | Path, parser: MultiDebugParser | None = None):
        self._root = Path(root)
        self._parser = parser or MultiDebugParser()

    def load(self, *, limit: int | None = None) -> list[RawBenchmarkCase]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        root = self._resolve_dataset_root(self._root)
        reference_directory = root / "2bug-MULTI-BUG" / "data_txts"
        if not reference_directory.is_dir():
            raise FileNotFoundError(
                "Could not find MULTI_DEBUG data_txts directory: "
                f"{reference_directory}"
            )

        cases: list[RawBenchmarkCase] = []
        for directory_name, bug_count in self._VARIANTS:
            directory = root / directory_name
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing MULTI_DEBUG directory: {directory}")

            for raw_path in sorted(directory.glob("*.py")):
                case = self._load_case(
                    raw_path=raw_path,
                    bug_count=bug_count,
                    reference_directory=reference_directory,
                )
                cases.append(case)
                if limit is not None and len(cases) >= limit:
                    return cases

        if not cases:
            raise ValueError("No MULTI_DEBUG cases were found")
        return cases

    def _load_case(
        self,
        *,
        raw_path: Path,
        bug_count: int,
        reference_directory: Path,
    ) -> RawBenchmarkCase:
        if bug_count == 1:
            reference_path = reference_directory / f"{raw_path.name}.txt"
            return self._parser.parse_one_bug(
                raw_text=raw_path.read_text(encoding="utf-8"),
                reference_text=reference_path.read_text(encoding="utf-8"),
                raw_path=raw_path,
                reference_path=reference_path,
            )

        processed_path = raw_path.parent / "data_txts" / f"{raw_path.name}.txt"
        return self._parser.parse_processed(
            text=processed_path.read_text(encoding="utf-8"),
            raw_path=raw_path,
            processed_path=processed_path,
            bug_count=bug_count,
        )

    @staticmethod
    def _resolve_dataset_root(root: Path) -> Path:
        if (root / "1bug-MULTI_BUG").is_dir():
            return root

        matches = list(root.rglob("1bug-MULTI_BUG"))
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely locate TreeInstruct Dataset under {root}")
        return matches[0].parent
