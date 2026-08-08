from __future__ import annotations

import os
from pathlib import Path

from app.agents.offline.offline_test_generator_agent import (
    OfflineTestGeneratorAgent,
)
from app.datasets.multi_debug_loader import (
    MultiDebugLoader,
)
from app.datasets.store import (
    BenchmarkCaseStore,
)
from app.execution.code_runner import (
    DockerCodeRunner,
)
from app.preprocessing.pipeline import (
    PreprocessingError,
    PreprocessingPipeline,
)
from dotenv import load_dotenv
from openai import OpenAI


class MultiDebugPreprocessingApp:
    """
    Loads the raw cases, generates tests for them, validates and saves everything to an output file
    """

    _loader: MultiDebugLoader
    _store: BenchmarkCaseStore
    _pipeline: PreprocessingPipeline

    def __init__(self):
        load_dotenv()
        self._loader = MultiDebugLoader(Path("data/raw/multi_debug"))
        self._store = BenchmarkCaseStore(Path("data/processed/multi_debug.jsonl"))
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        test_generator = OfflineTestGeneratorAgent(
            llm=client,
            model=os.environ["DATA_LLM_MODEL"],
        )
        self._pipeline = PreprocessingPipeline(
            test_generator=(test_generator),
            code_runner=(DockerCodeRunner()),
            max_attempts=3,
        )

    def run(self) -> None:
        raw_cases = self._loader.load(
            limit=self._max_cases()
        )  # parse the raw .py and .txt files
        completed = self._store.completed_case_ids()  # Loads the case_ids of the examples that were already processed. Useful for resuming processing
        for index, raw_case in enumerate(
            raw_cases,
            start=1,
        ):
            if raw_case.case_id in completed:
                continue

            print(f"[{index}/{len(raw_cases)}] {raw_case.case_id}")

            try:
                case = self._pipeline.process(
                    raw_case
                )  # I.e. generate tests and check that the correct code passes them and buggy one fails
            except PreprocessingError as error:
                print(f"FAILED: {error}")
                continue

            self._store.append(case)
            print("accepted")

    @staticmethod
    def _max_cases() -> int | None:
        value = os.getenv(
            "MAX_CASES",
            "",
        ).strip()
        return int(value) if value else None


if __name__ == "__main__":
    MultiDebugPreprocessingApp().run()
