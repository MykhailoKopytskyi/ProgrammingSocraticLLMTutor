from __future__ import annotations

from ...agents.offline.offline_bug_annotation_agent import (
    OfflineBugAnnotationAgent,
)
from ...common.code_runner import CodeRunner
from ...common.models import BenchmarkCase
from ..raw_benchmark_case import RawBenchmarkCase


class PreprocessingError(RuntimeError):
    pass


class QuixBugsCasePreprocessor:
    def __init__(
        self,
        *,
        bug_annotator: OfflineBugAnnotationAgent,
        code_runner: CodeRunner,
    ):
        self._bug_annotator = bug_annotator
        self._code_runner = code_runner

    def process(
        self,
        case: RawBenchmarkCase,
    ) -> BenchmarkCase:
        if not case.tests.strip():
            raise PreprocessingError(f"{case.case_id}: no tests were supplied")

        correct_run = self._code_runner.run(
            code=case.correct_code,
            tests=case.tests,
        )

        if not correct_run.passed:
            raise PreprocessingError(
                f"{case.case_id}: correct code failed\n{correct_run.output}"
            )

        buggy_run = self._code_runner.run(
            code=case.buggy_code,
            tests=case.tests,
        )

        if buggy_run.passed:
            raise PreprocessingError(
                f"{case.case_id}: buggy code passed the official tests"
            )

        bugs = self._bug_annotator.generate(
            problem_statement=case.problem_statement,
            buggy_code=case.buggy_code,
            correct_code=case.correct_code,
            expected_bug_count=1,
        )

        if len(bugs) != 1:
            raise PreprocessingError(
                f"{case.case_id}: expected exactly "
                "one bug annotation, "
                f"received {len(bugs)}"
            )

        observed_failure = buggy_run.output.strip()

        if not observed_failure:
            observed_failure = (
                f"Test execution failed with exit code {buggy_run.exit_code}"
            )

        return BenchmarkCase(
            case_id=case.case_id,
            problem_statement=case.problem_statement,
            buggy_code=case.buggy_code,
            tests=case.tests,
            observed_failure=observed_failure,
            bugs=list(bugs),
            correct_code=case.correct_code,
            source=case.source,
        )
