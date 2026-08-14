from __future__ import annotations

import random

from ...common.benchmark_case_store import BenchmarkCaseStore
from ..raw_benchmark_case import RawBenchmarkCase
from .case_preprocessor import (
    PreprocessingError,
    QuixBugsCasePreprocessor,
)
from .downloader import QuixBugsDownloader
from .loader import QuixBugsLoader


class QuixBugsPreprocessor:
    def __init__(
        self,
        *,
        downloader: QuixBugsDownloader,
        loader: QuixBugsLoader,
        case_preprocessor: QuixBugsCasePreprocessor,
        stores: dict[str, BenchmarkCaseStore],
        limit: int | None = None,
    ):
        self._downloader = downloader
        self._loader = loader
        self._case_preprocessor = case_preprocessor
        self._stores = stores
        self._limit = limit

    def preprocess(self) -> None:
        self._downloader.ensure_available()
        raw_cases = self._loader.load(limit=self._limit)
        splits = self._split(raw_cases)

        for split_name, cases in splits.items():
            self._process_split(
                split_name,
                cases,
            )

    def _process_split(
        self,
        split_name: str,
        cases: list[RawBenchmarkCase],
    ) -> None:
        if not cases:
            return

        store = self._stores[split_name]
        completed = store.completed_case_ids()
        pending = []

        for case in cases:
            if case.case_id not in completed:
                pending.append(case)

        if not pending:
            print(f"QuixBugs {split_name} is already preprocessed")
            return

        for index, raw_case in enumerate(
            pending,
            start=1,
        ):
            print(f"[QuixBugs {split_name} {index}/{len(pending)}] {raw_case.case_id}")

            try:
                case = self._case_preprocessor.process(raw_case)
            except PreprocessingError as error:
                print(f"FAILED: {error}")
                continue

            store.append(case)
            completed.add(case.case_id)

            print("accepted")

    @staticmethod
    def _split(
        cases: list[RawBenchmarkCase],
    ) -> dict[str, list[RawBenchmarkCase]]:
        cases = list(cases)

        rng = random.Random(42)
        rng.shuffle(cases)
        train_end = int(len(cases) * 0.8)
        validation_end = train_end + int(len(cases) * 0.1)

        return {
            "train": cases[:train_end],
            "validation": cases[train_end:validation_end],
            "test": cases[validation_end:],
        }
