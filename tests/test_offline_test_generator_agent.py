from types import SimpleNamespace

from app.agents.offline.offline_test_generator_agent import OfflineTestGeneratorAgent
from app.common.models import BugAnnotation, GeneratedTest, GeneratedTests
from app.datasets.base import RawBenchmarkCase


def make_raw_case() -> RawBenchmarkCase:
    return RawBenchmarkCase(
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
                fix="Use a stop argument one greater than the endpoint.",
            ),
        ),
        source="MULTI_DEBUG:range_case.py",
    )


class FakeResponses:
    def parse(self, *, model, instructions, input, text_format):
        assert text_format is GeneratedTests
        assert "ORIGINAL BUGGY CODE" in input
        assert "TRUSTED REFERENCE CORRECT CODE" in input
        assert "bug_1" in input
        assert "stop + 1" in input

        return SimpleNamespace(
            output_parsed=GeneratedTests(
                imports_and_fixtures="from solution import numbers",
                tests=[
                    GeneratedTest(
                        test_id="test_includes_endpoint",
                        test_code=(
                            "def test_includes_endpoint():\n"
                            "    assert numbers(1, 3) == [1, 2, 3]"
                        ),
                        purpose="Check that the final endpoint is included.",
                        related_bug_ids=["bug_1"],
                    )
                ],
            ),
            output_text="",
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_test_generator_returns_renderable_pytest():
    agent = OfflineTestGeneratorAgent(
        llm=FakeClient(),
        model="fake-test-model",
    )

    generated = agent.generate_tests(case=make_raw_case())
    pytest_file = generated.to_pytest_file()

    assert "from solution import numbers" in pytest_file
    assert "def test_includes_endpoint" in pytest_file
    assert pytest_file.endswith("\n")
