from __future__ import annotations

import os
from pathlib import Path

from app.agents.offline.offline_test_generator_agent import OfflineTestGeneratorAgent
from app.common.benchmark_case_store import BenchmarkCaseStore
from app.common.code_runner import DockerCodeRunner
from app.common.config import CONFIG
from app.preprocessing.multi_debug.case_preprocessor import MultiDebugCasePreprocessor
from app.preprocessing.multi_debug.downloader import MultiDebugDownloader
from app.preprocessing.multi_debug.loader import MultiDebugLoader
from app.preprocessing.multi_debug.preprocessor import MultiDebugPreprocessor
from dotenv import load_dotenv
from openai import OpenAI


class PreprocessingApp:
    def __init__(self):
        load_dotenv()
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        downloader = MultiDebugDownloader()
        test_generator = OfflineTestGeneratorAgent(
            llm=client,
            model=os.environ["DATA_LLM_MODEL"],
        )
        case_preprocessor = MultiDebugCasePreprocessor(
            test_generator=test_generator,
            code_runner=DockerCodeRunner(),
            max_attempts=CONFIG["OFFLINE_TEST_GENERATOR_MAX_ATTEMPTS"],
        )

        multi_debug = MultiDebugPreprocessor(
            downloader=downloader,
            loader=MultiDebugLoader(downloader.OUTPUT_DIRECTORY),
            case_preprocessor=case_preprocessor,
            store=BenchmarkCaseStore(Path("data/processed/multi_debug.jsonl")),
            limit=self._max_cases(),
        )
        self._preprocessors = (multi_debug,)

    def run(self) -> None:
        for preprocessor in self._preprocessors:
            preprocessor.preprocess()

    @staticmethod
    def _max_cases() -> int | None:
        value = os.getenv("MAX_CASES", "").strip()
        return int(value) if value else None


if __name__ == "__main__":
    PreprocessingApp().run()
