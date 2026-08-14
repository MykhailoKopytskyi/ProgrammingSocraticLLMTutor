from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from ...common.benchmark_case_store import BenchmarkCaseStore
from .case_preprocessor import PyMetaCaseError, PyMetaCasePreprocessor


class PyMetaPreprocessor:
    def __init__(
        self,
        *,
        case_preprocessor: PyMetaCasePreprocessor,
        store: BenchmarkCaseStore,
        csv_path: str | Path,
        limit: int | None = None,
    ):
        self._case_preprocessor = case_preprocessor
        self._store = store
        self._csv_path = str(csv_path)
        self._limit = limit

    def preprocess(self) -> None:
        dataset = load_dataset(
            "csv",
            data_files=self._csv_path,
            split="train",
        )
        completed = self._store.completed_case_ids()
        attempted = 0

        for record in dataset:
            if record["error_category"].strip().lower() != "error":
                continue

            case_id = self._case_preprocessor.case_id(record)

            if case_id in completed:
                continue
            if self._limit is not None and attempted >= self._limit:
                return

            attempted += 1
            print(f"processing: {case_id}")

            try:
                case = self._case_preprocessor.process(record)
            except PyMetaCaseError as error:
                print(f"FAILED: {error}")
                continue

            self._store.append(case)
            completed.add(case.case_id)
            print(f"accepted: {case.case_id}")
