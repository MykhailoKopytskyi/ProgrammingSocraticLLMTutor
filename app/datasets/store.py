from __future__ import annotations

from pathlib import Path

from ..common.models import BenchmarkCase


class BenchmarkCaseStore:
    """
    Responsible for writing the BenchmarkCase to the output file.
    Also, responsible for loading the caseIDs of the cases that were already processed
    """

    def __init__(
        self,
        path: str | Path,
    ):
        self._path = Path(path)

    def append(
        self,
        case: BenchmarkCase,
    ) -> None:
        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(case.model_dump_json())
            file.write("\n")

    def load(
        self,
    ) -> list[BenchmarkCase]:
        if not self._path.exists():
            return []

        with self._path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return [
                BenchmarkCase.model_validate_json(line) for line in file if line.strip()
            ]

    def completed_case_ids(
        self,
    ) -> set[str]:
        return {case.case_id for case in self.load()}
