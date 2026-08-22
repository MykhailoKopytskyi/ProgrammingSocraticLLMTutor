from __future__ import annotations

from typing import Any

from ...common.config import OFFLINE_PLANNER_INSTRUCTIONS
from ...common.models import BenchmarkCase, PedagogicalPlan, StrictModel
from ..agent import Agent


class OfflinePlannerOutput(StrictModel):
    """Oracle-assisted output generated only during offline data construction."""

    plan: PedagogicalPlan


class OfflinePlannerAgent(Agent):
    """
    Creates oracle-informed diagnoses and pedagogical plans for training data.
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
            reasoning_effort="medium",
        )

    def generate_output(
        self,
        case: BenchmarkCase,
        regeneration_feedback: str = "",
    ) -> OfflinePlannerOutput:
        prompt = (
            "Create a pedagogical plan for this Python debugging case. "
            "The training-only oracle contains the authoritative bug diagnosis, "
            "required fixes, and corrected code. Use them as grounding; do not "
            "invent or regenerate alternative bugs or repairs.\n\n"
            "RUNTIME-VISIBLE CONTEXT:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE CONTEXT:\n"
            f"{case.oracle_context()}"
        )

        if regeneration_feedback.strip():
            prompt += (
                "\n\nThe previous planner output was rejected by the oracle "
                "verifier. Regenerate complete plan while "
                "correcting these issues:\n"
                f"{regeneration_feedback.strip()}"
            )

        return self._get_structured_output(
            prompt=prompt,
            output_type=OfflinePlannerOutput,
        )
