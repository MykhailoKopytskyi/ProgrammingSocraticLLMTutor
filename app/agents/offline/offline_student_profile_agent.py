from __future__ import annotations

from typing import Any

from ...common.config import OFFLINE_STUDENT_PROFILE_INSTRUCTIONS
from ...common.models import BenchmarkCase
from ..agent import Agent
from .offline_student_agent import StudentProfile


class OfflineStudentProfileAgent(Agent):
    """Creates the private persistent misconception profile for one case."""

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_STUDENT_PROFILE_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
        )

    def generate(
        self,
        case: BenchmarkCase,
    ) -> StudentProfile:
        prompt = (
            "Generate the private student belief profile.\n\n"
            "RUNTIME-VISIBLE CASE:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE:\n"
            f"{case.oracle_context()}"
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=StudentProfile,
        )
