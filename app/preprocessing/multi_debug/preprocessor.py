from __future__ import annotations

from ...common.benchmark_case_store import BenchmarkCaseStore
from .case_preprocessor import MultiDebugCasePreprocessor, PreprocessingError
from .downloader import MultiDebugDownloader
from .loader import MultiDebugLoader


class MultiDebugPreprocessor:
    """Coordinates the complete MULTI_DEBUG preprocessing workflow."""

    def __init__(
        self,
        *,
        downloader: MultiDebugDownloader,
        loader: MultiDebugLoader,
        case_preprocessor: MultiDebugCasePreprocessor,
        store: BenchmarkCaseStore,
        limit: int | None = None,
    ):
        self._downloader = downloader
        self._loader = loader
        self._case_preprocessor = case_preprocessor
        self._store = store
        self._limit = limit

    def preprocess(self) -> None:
        self._downloader.ensure_available()
        raw_cases = self._loader.load(limit=self._limit)
        completed = self._store.completed_case_ids()

        pending = []
        for case in raw_cases:
            if case.case_id not in completed:
                pending.append(case)

        if not pending:
            print("MULTI_DEBUG is already preprocessed")
            return

        for index, raw_case in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {raw_case.case_id}")
            try:
                case = self._case_preprocessor.process(raw_case)
            except PreprocessingError as error:
                print(f"FAILED: {error}")
                continue

            self._store.append(case)
            print("accepted")
