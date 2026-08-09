from __future__ import annotations

from ..agents.offline.offline_test_generator_agent import OfflineTestGeneratorAgent
from ..common.models import BenchmarkCase
from ..datasets.base import RawBenchmarkCase
from ..execution.code_runner import CodeRunner


class PreprocessingError(RuntimeError):
    pass


class PreprocessingPipeline:
    """
    Responsible for processing the raw benchmark case object, i.e. generating tests.
    """

    _test_generator: OfflineTestGeneratorAgent
    _code_runner: CodeRunner
    _max_attempts: int

    def __init__(
        self,
        *,
        test_generator: OfflineTestGeneratorAgent,
        code_runner: CodeRunner,
        max_attempts: int = 3,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._test_generator = test_generator
        self._code_runner = code_runner
        self._max_attempts = max_attempts

    def process(
        self,
        case: RawBenchmarkCase,
    ) -> BenchmarkCase:
        feedback = ""

        for _ in range(self._max_attempts):
            generated_tests = self._test_generator.generate_tests(
                case=case,
                regeneration_feedback=feedback,
            )

            expected_bug_ids = {bug.bug_id for bug in case.bugs}
            covered_bug_ids = {
                bug_id
                for test in generated_tests.tests
                for bug_id in test.related_bug_ids
            }
            missing_bug_ids = expected_bug_ids - covered_bug_ids
            unsupported_bug_ids = covered_bug_ids - expected_bug_ids

            if missing_bug_ids or unsupported_bug_ids:
                feedback = (
                    "The generated test metadata does not cover the supplied "
                    "oracle bugs correctly. "
                    f"Missing bug IDs: {sorted(missing_bug_ids)}. "
                    f"Unsupported bug IDs: {sorted(unsupported_bug_ids)}."
                )
                continue

            tests = generated_tests.to_pytest_file()

            correct_execution = self._code_runner.run(
                code=case.correct_code,
                tests=tests,
            )

            if not correct_execution.passed:
                feedback = (
                    "The generated tests reject the trusted reference solution. "
                    "Revise the tests.\n\n"
                    f"{correct_execution.output}"
                )
                continue

            buggy_execution = self._code_runner.run(
                code=case.buggy_code,
                tests=tests,
            )

            if buggy_execution.passed:
                feedback = (
                    "The tests do not expose a failure in the buggy program. "
                    "Strengthen the tests so the annotated buggy behaviour fails."
                )
                continue

            return BenchmarkCase(
                case_id=case.case_id,
                problem_statement=case.problem_statement,
                buggy_code=case.buggy_code,
                tests=tests,
                observed_failure=buggy_execution.output,
                bugs=list(case.bugs),
                correct_code=case.correct_code,
                source=case.source,
            )

        raise PreprocessingError(
            f"Could not preprocess case {case.case_id!r} after "
            f"{self._max_attempts} attempts."
        )
