from __future__ import annotations

import os
from pathlib import Path

from app.agents.offline.offline_bug_annotation_agent import OfflineBugAnnotationAgent
from app.agents.offline.offline_problem_translation_agent import (
    OfflineProblemTranslationAgent,
)
from app.agents.offline.offline_pymeta_case_verifier_agent import (
    OfflinePyMetaCaseVerifierAgent,
)
from app.agents.offline.offline_reference_repair_agent import (
    OfflineReferenceRepairAgent,
)
from app.agents.offline.offline_test_generator_agent import OfflineTestGeneratorAgent
from app.common.benchmark_case_store import BenchmarkCaseStore
from app.common.code_runner import DockerCodeRunner
from app.common.config import CONFIG
from app.preprocessing.intro_prog.case_preprocessor import IntroProgCasePreprocessor
from app.preprocessing.intro_prog.preprocessor import IntroProgPreprocessor
from app.preprocessing.multi_debug.case_preprocessor import MultiDebugCasePreprocessor
from app.preprocessing.multi_debug.downloader import MultiDebugDownloader
from app.preprocessing.multi_debug.loader import MultiDebugLoader
from app.preprocessing.multi_debug.preprocessor import MultiDebugPreprocessor
from app.preprocessing.pymeta.case_preprocessor import PyMetaCasePreprocessor
from app.preprocessing.pymeta.preprocessor import PyMetaPreprocessor
from app.preprocessing.quixbugs.case_preprocessor import (
    QuixBugsCasePreprocessor,
)
from app.preprocessing.quixbugs.downloader import QuixBugsDownloader
from app.preprocessing.quixbugs.loader import QuixBugsLoader
from app.preprocessing.quixbugs.preprocessor import QuixBugsPreprocessor
from dotenv import load_dotenv
from openai import OpenAI


class PreprocessingApp:
    def __init__(self):
        load_dotenv()
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Multi debug
        multi_debug_downloader = MultiDebugDownloader()
        test_generator = OfflineTestGeneratorAgent(
            llm=client,
            model=os.environ["DATA_LLM_MODEL"],
        )
        multi_debug_case_preprocessor = MultiDebugCasePreprocessor(
            test_generator=test_generator,
            code_runner=DockerCodeRunner(),
            max_attempts=CONFIG["OFFLINE_TEST_GENERATOR_MAX_ATTEMPTS"],
        )

        multi_debug = MultiDebugPreprocessor(
            downloader=multi_debug_downloader,
            loader=MultiDebugLoader(multi_debug_downloader.OUTPUT_DIRECTORY),
            case_preprocessor=multi_debug_case_preprocessor,
            store=BenchmarkCaseStore(Path("data/processed/multi_debug.jsonl")),
            limit=self._max_cases(),
        )

        # IntroProg
        intro_prog_bug_annotator = OfflineBugAnnotationAgent(
            llm=client,
            model=os.environ["DATA_LLM_MODEL"],
        )
        intro_prog_case_preprocessor = IntroProgCasePreprocessor(
            bug_annotator=intro_prog_bug_annotator,
            code_runner=DockerCodeRunner(),
        )

        intro_prog = IntroProgPreprocessor(
            case_preprocessor=intro_prog_case_preprocessor,
            stores={
                "train": BenchmarkCaseStore("data/processed/intro_prog/train.jsonl"),
                "validation": BenchmarkCaseStore(
                    "data/processed/intro_prog/validation.jsonl"
                ),
                "test": BenchmarkCaseStore("data/processed/intro_prog/test.jsonl"),
            },
            limit=self._max_cases(),
        )

        # Quixbugs
        quixbugs_bug_annotator = OfflineBugAnnotationAgent(
            llm=client,
            model=os.environ["DATA_LLM_MODEL"],
        )
        quixbugs_case_preprocessor = QuixBugsCasePreprocessor(
            bug_annotator=quixbugs_bug_annotator,
            code_runner=DockerCodeRunner(),
        )
        quixbugs_downloader = QuixBugsDownloader()
        quixbugs = QuixBugsPreprocessor(
            downloader=quixbugs_downloader,
            loader=QuixBugsLoader(quixbugs_downloader.OUTPUT_DIRECTORY),
            case_preprocessor=quixbugs_case_preprocessor,
            stores={
                "train": BenchmarkCaseStore("data/processed/quixbugs/train.jsonl"),
                "validation": BenchmarkCaseStore(
                    "data/processed/quixbugs/validation.jsonl"
                ),
                "test": BenchmarkCaseStore("data/processed/quixbugs/test.jsonl"),
            },
            limit=self._max_cases(),
        )

        # Pymeta
        pymeta_translator = OfflineProblemTranslationAgent(
            llm=client,
            model=os.environ["PYMETA_TRANSLATION_MODEL"],
            reasoning_effort=(
                os.getenv("PYMETA_TRANSLATION_REASONING_EFFORT", "").strip() or None
            ),
        )

        pymeta_repair_agent = OfflineReferenceRepairAgent(
            llm=client,
            model=os.environ["PYMETA_REPAIR_MODEL"],
            reasoning_effort=(
                os.getenv("PYMETA_REPAIR_REASONING_EFFORT", "").strip() or None
            ),
        )

        pymeta_bug_annotator = OfflineBugAnnotationAgent(
            llm=client,
            model=os.environ["PYMETA_BUG_ANNOTATION_MODEL"],
            reasoning_effort=(
                os.getenv("PYMETA_BUG_ANNOTATION_REASONING_EFFORT", "").strip() or None
            ),
        )

        pymeta_test_generator = OfflineTestGeneratorAgent(
            llm=client,
            model=os.environ["PYMETA_TEST_MODEL"],
            reasoning_effort=(
                os.getenv("PYMETA_TEST_REASONING_EFFORT", "").strip() or None
            ),
        )

        pymeta_case_verifier = OfflinePyMetaCaseVerifierAgent(
            llm=client,
            model=os.environ["PYMETA_VERIFIER_MODEL"],
            reasoning_effort=(
                os.getenv("PYMETA_VERIFIER_REASONING_EFFORT", "").strip() or None
            ),
        )

        pymeta_case_preprocessor = PyMetaCasePreprocessor(
            translator=pymeta_translator,
            bug_annotator=pymeta_bug_annotator,
            test_generator=pymeta_test_generator,
            case_verifier=pymeta_case_verifier,
            repair_agent=pymeta_repair_agent,
            code_runner=DockerCodeRunner(),
            test_max_attempts=1,
            verification_max_attempts=3,
        )
        pymeta = PyMetaPreprocessor(
            case_preprocessor=pymeta_case_preprocessor,
            store=BenchmarkCaseStore(Path("data/processed/pymeta.jsonl")),
            csv_path="data/raw/pymeta/pymeta_best_400.csv",
            limit=self._max_cases(),
        )

        self._preprocessors = (multi_debug, intro_prog, quixbugs, pymeta)

    def run(self) -> None:
        for preprocessor in self._preprocessors:
            preprocessor.preprocess()

    @staticmethod
    def _max_cases() -> int | None:
        value = os.getenv("MAX_CASES", "").strip()
        return int(value) if value else None


if __name__ == "__main__":
    PreprocessingApp().run()
