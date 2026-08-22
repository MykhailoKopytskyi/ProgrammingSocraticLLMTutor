from __future__ import annotations

from typing import Any

from pydantic import Field

from ...common.code_runner import TestRunResult
from ...common.config import OFFLINE_TUTOR_TURN_VERIFIER_INSTRUCTIONS
from ...common.models import (
    BenchmarkCase,
    PedagogicalPlan,
    PlanProgress,
    StrictModel,
)
from ..agent import Agent
from .offline_tutor_agent import TutorTurn


class OfflineTutorTurnVerifierAgent(Agent):
    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_TUTOR_TURN_VERIFIER_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
            reasoning_effort="medium",
            max_output_tokens=3000,
        )

    def verify(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        candidate: TutorTurn,
        verified_history: str,
        latest_code_execution: TestRunResult | None = None,
        previous_regeneration_feedback: str = "",
    ) -> TutorHardCheck:
        execution_evidence = (
            "No student-submitted code has been executed yet."
            if latest_code_execution is None
            else (
                f"passed={latest_code_execution.passed}\n{latest_code_execution.output}"
            )
        )

        prompt = (
            "Evaluate the proposed Tutor turn before it is added to the "
            "conversation.\n\n"
            "RUNTIME-VISIBLE CASE:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE:\n"
            f"{case.oracle_context()}\n\n"
            "FIXED PEDAGOGICAL PLAN:\n"
            f"{plan.model_dump_json(indent=2)}\n\n"
            "PRE-TURN PLAN PROGRESS:\n"
            f"{progress.model_dump_json(indent=2)}\n\n"
            "LATEST STUDENT CODE EXECUTION:\n"
            f"{execution_evidence}\n\n"
            "ACCEPTED CONVERSATION HISTORY:\n"
            f"{verified_history}\n\n"
            "CANDIDATE TUTOR TURN:\n"
            f"{candidate.model_dump_json(indent=2)}"
            "PREVIOUS REGENERATION FEEDBACK GIVEN TO THIS TUTOR:\n"
            f"{previous_regeneration_feedback or '[none]'}\n\n"
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=TutorHardCheck,
        )


class TutorHardCheck(StrictModel):
    technical_error: bool
    learner_state_mismatch: bool
    wrong_active_step: bool
    unjustified_step_completion: bool
    latest_student_turn_not_addressed: bool
    solution_leakage: bool
    malformed_or_incoherent: bool
    serious_repetition: bool
    missed_step_completion: bool

    reasons: list[str] = Field(default_factory=list)
    regeneration_feedback: str | None = None

    @property
    def accepted(self) -> bool:
        return not any(
            (
                self.technical_error,
                self.wrong_active_step,
                self.unjustified_step_completion,
                self.latest_student_turn_not_addressed,
                self.solution_leakage,
                self.malformed_or_incoherent,
                self.serious_repetition,
                self.learner_state_mismatch,
                self.missed_step_completion,
            )
        )
