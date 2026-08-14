from __future__ import annotations

from typing import Any

from ...common.config import OFFLINE_PLAN_VERIFIER_INSTRUCTIONS
from ...common.models import BenchmarkCase, StrictModel
from ..agent import Agent
from .offline_planner_agent import OfflinePlannerOutput


class OfflinePlanVerifierAgent(Agent):
    """Checks an offline candidate diagnosis and plan against training-only ground truth."""

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_PLAN_VERIFIER_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
        )

    def verify(
        self,
        case: BenchmarkCase,
        planner_output: OfflinePlannerOutput,
    ) -> PlanVerification:
        prompt = (
            "Verify the candidate pedagogical plan.\n\n"
            "RUNTIME-VISIBLE CONTEXT:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE CONTEXT:\n"
            f"{case.oracle_context()}\n\n"
            "CANDIDATE OFFLINE PLANNER OUTPUT:\n"
            f"{planner_output.model_dump_json(indent=2)}"
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=PlanVerification,
        )


class PlanVerification(StrictModel):
    accepted: bool

    covered_bug_ids: list[str]
    missing_bug_ids: list[str]
    invented_or_unsupported_claims: list[str]
    errors: list[str]

    regeneration_feedback: str
