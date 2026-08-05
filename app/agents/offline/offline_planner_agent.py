from __future__ import annotations

from typing import Any

from ...common.config import OFFLINE_PLANNER_INSTRUCTIONS
from ...common.models import BenchmarkCase, PlannerOutput
from ..agent import Agent


class OfflinePlannerAgent(Agent):
    """
    Creates oracle-informed pedagogical plans for training data generation.
    This agent is never used in a real tutoring session.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_PLANNER_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
        )

    def generate_output(
        self,
        case: BenchmarkCase,
        regeneration_feedback: str = "",
    ) -> PlannerOutput:
        prompt = (
            "Diagnose, repair, and create a pedagogical plan for this Python "
            "debugging case.\n\n"
            "RUNTIME-VISIBLE CONTEXT:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE CONTEXT:\n"
            f"{case.oracle_context()}"
        )

        if regeneration_feedback.strip():
            prompt += (
                "The previous planner output was rejected by the oracle verifier. "
                "Regenerate the diagnosis, corrected code, and complete plan while "
                "correcting these issues:\n"
                f"{regeneration_feedback.strip()}"
            )

        return self._get_structured_output(
            prompt=prompt,
            output_type=PlannerOutput,
        )
