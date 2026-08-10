from __future__ import annotations

from typing import Any

from ...common.config import (
    REFERENCE_REPAIR_AGENT_INSTRUCTIONS,
)
from ...common.models import ReferenceRepair
from ...preprocessing.raw_benchmark_case import RawBenchmarkCase
from ..agent import Agent


class OfflineReferenceRepairAgent(Agent):
    """
    Fallback for preprocessing datasets that do not provide trusted corrected code.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = REFERENCE_REPAIR_AGENT_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
        )

    def generate_repair(
        self,
        case: RawBenchmarkCase,
        regeneration_feedback: str = "",
    ) -> ReferenceRepair:
        bug_lines = []
        for bug in case.bugs:
            bug_lines.append(
                f"{bug.bug_id}\nDescription: {bug.description}\nRequired fix: {bug.fix}"
            )
        oracle_bugs = "\n".join(bug_lines)

        prompt = (
            "Create the minimal corrected program.\n\n"
            "PROBLEM STATEMENT:\n"
            f"{case.problem_statement.strip()}\n\n"
            "ORIGINAL BUGGY CODE:\n"
            f"{case.buggy_code.rstrip()}\n\n"
            "TRAINING-ONLY BUGS AND REQUIRED FIXES:\n"
            f"{oracle_bugs}"
        )

        if regeneration_feedback.strip():
            prompt += f"\n\nREGENERATION FEEDBACK:\n{regeneration_feedback.strip()}"

        return self._get_structured_output(
            prompt=prompt,
            output_type=ReferenceRepair,
        )
