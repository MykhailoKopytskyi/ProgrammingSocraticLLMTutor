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
    Generates a minimal student-aligned repair during dataset preprocessing.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        reasoning_effort: str | None = None,
        instructions: str = REFERENCE_REPAIR_AGENT_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            reasoning_effort=reasoning_effort,
            instructions=instructions,
        )

    def generate_repair(
        self,
        case: RawBenchmarkCase,
        regeneration_feedback: str = "",
        independent_reference: bool = False,
    ) -> ReferenceRepair:
        if independent_reference:
            prompt = (
                "Create a correct version of the student's program using the "
                "independent reference as a correctness oracle.\n\n"
                "PROBLEM STATEMENT:\n"
                f"{case.problem_statement.strip()}\n\n"
                "ORIGINAL STUDENT CODE:\n"
                f"{case.buggy_code.rstrip()}\n\n"
                "TRUSTED INDEPENDENT REFERENCE SOLUTION:\n"
                f"{case.correct_code.rstrip()}"
            )
        else:
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
