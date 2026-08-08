from app.common.models import BugAnnotation, GeneratedTest, GeneratedTests
from app.datasets.base import RawBenchmarkCase
from app.execution.code_runner import TestRunResult
from app.preprocessing.pipeline import PreprocessingPipeline


class FakeTestGenerator:
    def generate_tests(self, *, case, regeneration_feedback=""):
        return GeneratedTests(
            imports_and_fixtures="from solution import numbers",
            tests=[
                GeneratedTest(
                    test_id="test_endpoint",
                    test_code=(
                        "def test_endpoint():\n    assert numbers(1, 3) == [1, 2, 3]"
                    ),
                    purpose="Check endpoint inclusion.",
                    related_bug_ids=["bug_1"],
                )
            ],
        )


class FakeRunner:
    def run(self, *, code, tests):
        if "stop + 1" in code:
            return TestRunResult(
                passed=True,
                exit_code=0,
                stdout="1 passed",
                stderr="",
            )

        return TestRunResult(
            passed=False,
            exit_code=1,
            stdout="1 failed",
            stderr="",
        )


def test_preprocessing_pipeline_builds_complete_benchmark_case():
    raw_case = RawBenchmarkCase(
        case_id="range-case",
        problem_statement="Return all integers from start through stop.",
        buggy_code="def numbers(start, stop):\n    return list(range(start, stop))\n",
        correct_code=(
            "def numbers(start, stop):\n    return list(range(start, stop + 1))\n"
        ),
        bugs=(
            BugAnnotation(
                bug_id="bug_1",
                description="range excludes its stop argument.",
                fix="Use a stop value one greater than the endpoint.",
            ),
        ),
        source="MULTI_DEBUG:range_case.py",
    )

    pipeline = PreprocessingPipeline(
        test_generator=FakeTestGenerator(),
        code_runner=FakeRunner(),
    )
    case = pipeline.process(raw_case)

    assert case.tests
    assert case.observed_failure == "1 failed"
    assert "stop + 1" in case.correct_code
