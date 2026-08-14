from __future__ import annotations

from typing import Any

from pydantic import Field

from app.common.config import OFFLINE_BUG_ANNOTATION_AGENT_INSTRUCTIONS

from ...common.models import BugAnnotation, StrictModel
from ..agent import Agent


class BugAnnotationOutput(StrictModel):
    bugs: list[BugAnnotation] = Field(min_length=1, max_length=5)


class OfflineBugAnnotationAgent(Agent):
    """Derives frozen bug annotations from a trusted buggy/corrected code pair."""

    def __init__(
        self,
        llm: Any,
        model: str,
        reasoning_effort: str | None = None,
        instructions: str = OFFLINE_BUG_ANNOTATION_AGENT_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            reasoning_effort=reasoning_effort,
            instructions=instructions,
        )

    def generate(
        self,
        *,
        problem_statement: str,
        buggy_code: str,
        correct_code: str,
        expected_bug_count: int | None = None,
        independent_reference: bool = False,
        regeneration_feedback: str = "",
    ) -> tuple[BugAnnotation, ...]:
        prompt = (
            "PROBLEM:\n"
            f"{problem_statement.strip()}\n\n"
            "BUGGY STUDENT CODE:\n"
            f"{buggy_code.rstrip()}\n\n"
            "TRUSTED CORRECTED CODE:\n"
            f"{correct_code.rstrip()}"
        )
        if regeneration_feedback.strip():
            prompt += f"\n\nREGENERATION FEEDBACK:\n{regeneration_feedback.strip()}"

        if independent_reference:
            prompt += """\n\nThe trusted code is an independent reference solution, not 
            a minimal repair of the student's program. Identify only actual 
            semantic defects in the student's code. Different structure, 
            variable names, algorithm choice or formatting are not bugs."""

        if expected_bug_count is not None:
            prompt += (
                "\n\nEXPECTED BUG COUNT:\n"
                f"There are exactly {expected_bug_count} intended "
                "semantic bugs in this case. Return exactly "
                f"{expected_bug_count} BugAnnotation objects."
            )
        output = self._get_structured_output(
            prompt=prompt,
            output_type=BugAnnotationOutput,
        )
        bugs = []
        for index, bug in enumerate(
            output.bugs,
            start=1,
        ):
            bugs.append(
                BugAnnotation(
                    bug_id=f"bug_{index}",
                    description=bug.description,
                    fix=bug.fix,
                )
            )

        return tuple(bugs)
