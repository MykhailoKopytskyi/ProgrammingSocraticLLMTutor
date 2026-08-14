from __future__ import annotations

import io
import re
import tokenize

from app.agents.offline.offline_pymeta_case_verifier_agent import (
    OfflinePyMetaCaseVerifierAgent,
)

from ...agents.offline.offline_bug_annotation_agent import OfflineBugAnnotationAgent
from ...agents.offline.offline_problem_translation_agent import (
    OfflineProblemTranslationAgent,
)
from ...agents.offline.offline_reference_repair_agent import (
    OfflineReferenceRepairAgent,
)
from ...agents.offline.offline_test_generator_agent import OfflineTestGeneratorAgent
from ...common.code_runner import CodeRunner, TestRunResult
from ...common.models import BenchmarkCase
from ..raw_benchmark_case import RawBenchmarkCase


class PyMetaCaseError(RuntimeError):
    pass


class PyMetaCasePreprocessor:
    def __init__(
        self,
        *,
        translator: OfflineProblemTranslationAgent,
        bug_annotator: OfflineBugAnnotationAgent,
        test_generator: OfflineTestGeneratorAgent,
        case_verifier: OfflinePyMetaCaseVerifierAgent,
        repair_agent: OfflineReferenceRepairAgent,
        code_runner: CodeRunner,
        test_max_attempts: int = 3,
        verification_max_attempts: int = 3,
    ):
        self._translator = translator
        self._bug_annotator = bug_annotator
        self._test_generator = test_generator
        self._case_verifier = case_verifier
        self._repair_agent = repair_agent
        self._code_runner = code_runner
        self._test_max_attempts = test_max_attempts
        self._verification_max_attempts = verification_max_attempts

    def process(self, record: dict) -> BenchmarkCase:
        case_id = self.case_id(record)
        original_problem = record["question"]
        buggy_code = self._clean_code(record["studentAnswer"])
        reference_code = self._clean_code(record["exceptedAnswer"])

        if self._contains_chinese(buggy_code):
            raise PyMetaCaseError(f"{case_id}: Chinese remains in student code")
        if self._contains_chinese(reference_code):
            raise PyMetaCaseError(f"{case_id}: Chinese remains in reference code")

        regenerate_from = "translation"
        feedback = ""

        for _ in range(self._verification_max_attempts):
            if regenerate_from == "translation":
                problem_statement = self._translator.translate(
                    original_problem,
                    feedback=feedback or None,
                )

                if self._contains_chinese(problem_statement):
                    feedback = (
                        "The previous translation still contains Chinese. "
                        "Translate all explanatory Chinese into English while "
                        "preserving the original programming requirements."
                    )
                    continue

                regenerate_from = "repair"
                feedback = ""

            if regenerate_from == "repair":
                reference_case = RawBenchmarkCase(
                    case_id=case_id,
                    problem_statement=problem_statement,
                    buggy_code=buggy_code,
                    correct_code=reference_code,
                    bugs=(),
                    source=f"PYMETA:{record['questionId']}",
                )

                repair = self._repair_agent.generate_repair(
                    case=reference_case,
                    regeneration_feedback=feedback,
                    independent_reference=True,
                )
                corrected_code = repair.corrected_code.strip("\n")

                if self._contains_chinese(corrected_code):
                    feedback = (
                        "The previous corrected program still contains Chinese. "
                        "Return an English-only student-aligned corrected program "
                        "without changing required behaviour."
                    )
                    continue

                regenerate_from = "annotations"
                feedback = ""

            if regenerate_from == "annotations":
                bugs = self._bug_annotator.generate(
                    problem_statement=problem_statement,
                    buggy_code=buggy_code,
                    correct_code=corrected_code,
                    regeneration_feedback=feedback,
                )

                annotation_has_chinese = False
                for bug in bugs:
                    if self._contains_chinese(bug.description):
                        annotation_has_chinese = True
                    if self._contains_chinese(bug.fix):
                        annotation_has_chinese = True

                if annotation_has_chinese:
                    feedback = (
                        "The previous bug annotations contain Chinese. "
                        "Return every description and fix in English."
                    )
                    continue

                regenerate_from = "tests"
                feedback = ""

            if regenerate_from == "tests":
                case = RawBenchmarkCase(
                    case_id=case_id,
                    problem_statement=problem_statement,
                    buggy_code=buggy_code,
                    correct_code=corrected_code,
                    bugs=bugs,
                    source=f"PYMETA:{record['questionId']}",
                )

                tests, buggy_run, corrected_run, reference_run = self._generate_tests(
                    case=case,
                    reference_code=reference_code,
                    regeneration_feedback=feedback,
                )

                if self._contains_chinese(tests):
                    feedback = (
                        "The previous generated tests contain Chinese. "
                        "Return English-only pytest code."
                    )
                    continue

                feedback = ""

            verification = self._case_verifier.verify(
                original_problem=original_problem,
                translated_problem=problem_statement,
                buggy_code=buggy_code,
                reference_code=reference_code,
                corrected_code=corrected_code,
                bugs=list(bugs),
                tests=tests,
                buggy_execution=(
                    f"passed={buggy_run.passed}, "
                    f"exit_code={buggy_run.exit_code}, "
                    f"timed_out={buggy_run.timed_out}\n"
                    f"{buggy_run.output}"
                ),
                corrected_execution=(
                    f"passed={corrected_run.passed}, "
                    f"exit_code={corrected_run.exit_code}, "
                    f"timed_out={corrected_run.timed_out}\n"
                    f"{corrected_run.output}"
                ),
                reference_execution=(
                    f"passed={reference_run.passed}, "
                    f"exit_code={reference_run.exit_code}, "
                    f"timed_out={reference_run.timed_out}\n"
                    f"{reference_run.output}"
                ),
            )

            if verification.accepted:
                return BenchmarkCase(
                    case_id=case.case_id,
                    problem_statement=case.problem_statement,
                    buggy_code=case.buggy_code,
                    tests=tests,
                    observed_failure=buggy_run.output,
                    bugs=list(case.bugs),
                    correct_code=case.correct_code,
                    source=case.source,
                )

            if verification.regenerate_from == "drop":
                raise PyMetaCaseError(
                    f"{case_id}: verifier requested drop: "
                    f"{verification.regeneration_feedback}"
                )

            if verification.regenerate_from == "none":
                raise PyMetaCaseError(
                    f"{case_id}: verifier rejected case without regeneration stage"
                )

            regenerate_from = verification.regenerate_from
            feedback = verification.regeneration_feedback or ""

            if not feedback:
                feedback = "; ".join(verification.reasons)

        raise PyMetaCaseError(
            f"{case_id}: case verifier rejected generated case after retries"
        )

    def _generate_tests(
        self,
        *,
        case: RawBenchmarkCase,
        reference_code: str,
        regeneration_feedback: str = "",
    ) -> tuple[
        str,
        TestRunResult,
        TestRunResult,
        TestRunResult,
    ]:
        generated = self._test_generator.generate_tests(
            case=case,
            regeneration_feedback=regeneration_feedback,
        )
        tests = generated.to_pytest_file()

        corrected_run = self._code_runner.run(
            code=case.correct_code,
            tests=tests,
        )
        reference_run = self._code_runner.run(
            code=reference_code,
            tests=tests,
        )
        buggy_run = self._code_runner.run(
            code=case.buggy_code,
            tests=tests,
        )

        return tests, buggy_run, corrected_run, reference_run

    @staticmethod
    def case_id(record: dict) -> str:
        return f"pymeta__{record['questionId']}__{record['attemptstepid']}"

    @classmethod
    def _clean_code(cls, code: str) -> str:
        code = code.replace("\r\n", "\n").replace("\r", "\n")
        lines = []

        for line in code.split("\n"):
            lines.append(cls._clean_line(line))

        return "\n".join(lines).strip("\n")

    @classmethod
    def _clean_line(cls, line: str) -> str:
        indentation = len(line) - len(line.lstrip())
        text = line.lstrip()

        if not text:
            return ""

        try:
            tokens = tokenize.generate_tokens(io.StringIO(text + "\n").readline)

            for token in tokens:
                if token.type != tokenize.COMMENT:
                    continue
                if not cls._contains_chinese(token.string):
                    continue

                end = indentation + token.start[1]
                return line[:end].rstrip()

        except (tokenize.TokenError, IndentationError, SyntaxError):
            return line

        return line

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text) is not None
