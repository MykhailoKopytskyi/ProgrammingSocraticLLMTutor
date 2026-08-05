from __future__ import annotations

from typing import Any

from ...common.config import OFFLINE_PLAN_VERIFIER_INSTRUCTIONS
from ...common.models import BenchmarkCase, PlannerOutput, StrictModel
from ..agent import Agent


class OfflinePlanVerifierAgent(Agent):
    """Checks a candidate plan against training only ground truth."""

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
        planner_output: PlannerOutput,
        observed_failure: str = "",
    ) -> PlanVerification:
        prompt = (
            "Verify the candidate diagnosis, corrected code, and pedagogical "
            "plan.\n\n"
            "RUNTIME-VISIBLE CONTEXT:\n"
            f"{case.visible_context(observed_failure)}\n\n"
            "TRAINING-ONLY ORACLE CONTEXT:\n"
            f"{case.oracle_context()}\n\n"
            "CANDIDATE PLANNER OUTPUT:\n"
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
