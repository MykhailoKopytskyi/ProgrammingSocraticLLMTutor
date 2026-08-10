from __future__ import annotations

from pathlib import Path

from .models import BenchmarkCase


class BenchmarkCaseStore:
    """Reads and writes processed benchmark cases as JSONL."""

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def append(self, case: BenchmarkCase) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(case.model_dump_json())
            file.write("\n")

    def load(self) -> list[BenchmarkCase]:
        if not self._path.exists():
            return []

        cases: list[BenchmarkCase] = []
        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    cases.append(BenchmarkCase.model_validate_json(line))
        return cases

    def completed_case_ids(self) -> set[str]:
        case_ids: set[str] = set()
        for case in self.load():
            case_ids.add(case.case_id)
        return case_ids
