from __future__ import annotations

from typing import Any

from ...common.config import OFFLINE_TEST_GENERATOR_INSTRUCTIONS
from ...common.models import GeneratedTests
from ...datasets.base import RawBenchmarkCase
from ..agent import Agent


class OfflineTestGeneratorAgent(Agent):
    """Generates candidate pytest tests during dataset preprocessing."""

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_TEST_GENERATOR_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
        )

    def generate_tests(
        self,
        *,
        case: RawBenchmarkCase,
        regeneration_feedback: str = "",
    ) -> GeneratedTests:
        oracle_bugs = "\n\n".join(
            f"{bug.bug_id}\nDescription: {bug.description}\nRequired fix: {bug.fix}"
            for bug in case.bugs
        )

        prompt = (
            "Generate a compact pytest suite for this case.\n\n"
            "PROBLEM STATEMENT:\n"
            f"{case.problem_statement.strip()}\n\n"
            "ORIGINAL BUGGY CODE:\n"
            f"{case.buggy_code.rstrip()}\n\n"
            "TRAINING-ONLY BUGS AND REQUIRED FIXES:\n"
            f"{oracle_bugs}\n\n"
            "TRUSTED REFERENCE CORRECT CODE:\n"
            f"{case.correct_code.rstrip()}"
        )

        if regeneration_feedback.strip():
            prompt += f"\n\nREGENERATION FEEDBACK:\n{regeneration_feedback.strip()}"

        return self._get_structured_output(
            prompt=prompt,
            output_type=GeneratedTests,
        )
