from __future__ import annotations

from typing import Any

from pydantic import Field

from ...common.models import BugAnnotation, StrictModel
from ..agent import Agent


class BugAnnotationOutput(StrictModel):
    bugs: list[BugAnnotation] = Field(min_length=1, max_length=3)


class OfflineBugAnnotationAgent(Agent):
    """Derives frozen bug annotations from a trusted buggy/corrected code pair."""

    def __init__(self, llm: Any, model: str):
        super().__init__(
            llm=llm,
            model=model,
            instructions=(
                "Identify only the bugs actually fixed between the student's buggy "
                "program and the trusted corrected program. Do not invent unrelated "
                "bugs. Return concise descriptions and fixes."
            ),
        )

    def generate(
        self,
        *,
        problem_statement: str,
        buggy_code: str,
        correct_code: str,
    ) -> tuple[BugAnnotation, ...]:
        output = self._get_structured_output(
            prompt=(
                "PROBLEM:\n"
                f"{problem_statement.strip()}\n\n"
                "BUGGY STUDENT CODE:\n"
                f"{buggy_code.rstrip()}\n\n"
                "TRUSTED CORRECTED CODE:\n"
                f"{correct_code.rstrip()}"
            ),
            output_type=BugAnnotationOutput,
        )

        return tuple(output.bugs)
