from types import SimpleNamespace

from app.agents.offline.offline_reference_repair_agent import (
    OfflineReferenceRepairAgent,
)
from app.common.models import AppliedFix, BugAnnotation, ReferenceRepair
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
        assert text_format is ReferenceRepair
        assert "ORIGINAL BUGGY CODE" in input
        assert "bug_1" in input

        return SimpleNamespace(
            output_parsed=ReferenceRepair(
                corrected_code=(
                    "def numbers(start, stop):\n"
                    "    return list(range(start, stop + 1))\n"
                ),
                applied_fixes=[
                    AppliedFix(
                        bug_id="bug_1",
                        explanation="Include the requested endpoint.",
                    )
                ],
            ),
            output_text="",
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_reference_repair_agent_can_still_be_used_for_other_datasets():
    agent = OfflineReferenceRepairAgent(
        llm=FakeClient(),
        model="fake-repair-model",
    )

    repair = agent.generate_repair(make_raw_case())

    assert "stop + 1" in repair.corrected_code
    assert repair.applied_fixes[0].bug_id == "bug_1"
