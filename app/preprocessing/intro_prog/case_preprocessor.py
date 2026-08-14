from textwrap import indent

from ...agents.offline.offline_bug_annotation_agent import OfflineBugAnnotationAgent
from ...common.code_runner import CodeRunner
from ...common.models import BenchmarkCase


class IntroProgCasePreprocessor:
    def __init__(
        self,
        *,
        bug_annotator: OfflineBugAnnotationAgent,
        code_runner: CodeRunner,
    ):
        self._bug_annotator = bug_annotator
        self._code_runner = code_runner

    def process(self, record: dict) -> BenchmarkCase | None:
        case_id = f"intro_prog__{record['submission_id']}"
        tests = self._make_tests(record["test"])

        correct_run = self._code_runner.run(
            code=record["annotation"],
            tests=tests,
        )
        if not correct_run.passed:
            print(f"{case_id}: CORRECT CODE FAILED")
            print(correct_run.output)
            return None

        buggy_run = self._code_runner.run(
            code=record["func_code"],
            tests=tests,
        )
        if buggy_run.passed:
            print(f"{case_id}: BUGGY CODE PASSED")
            return None

        bugs = self._bug_annotator.generate(
            problem_statement=record["description"],
            buggy_code=record["func_code"],
            correct_code=record["annotation"],
        )

        return BenchmarkCase(
            case_id=case_id,
            problem_statement=record["description"],
            buggy_code=record["func_code"],
            tests=tests,
            observed_failure=buggy_run.output,
            bugs=list(bugs),
            correct_code=record["annotation"],
            source=f"INTRO_PROG:{record['assignment_id']}",
        )

    @staticmethod
    def _make_tests(test: str) -> str:
        return (
            "from solution import *\n\n"
            "def test_solution():\n"
            f"{indent(test.strip(), '    ')}\n"
        )
