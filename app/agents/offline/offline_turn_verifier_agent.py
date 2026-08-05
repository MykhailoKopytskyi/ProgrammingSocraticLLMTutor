from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...common.config import OFFLINE_TURN_VERIFIER_INSTRUCTIONS
from ...common.message import Message
from ...common.models import BenchmarkCase, PedagogicalPlan, PlanProgress, StrictModel
from ..agent import Agent
from ..tutor_agent import TutorTurn


class OfflineTurnVerifierAgent(Agent):
    llm: Any
    model: str
    instructions: str

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_TURN_VERIFIER_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
        )

    def verify(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        history: list[Message],
        candidate: TutorTurn,
    ) -> TutorHardCheck:
        if not history:
            raise ValueError("OfflineTurnVerifierAgent requires conversation history")

        if history[-1]["role"] != "student":
            raise ValueError(
                "History must end with the Student message to which the "
                "candidate responds"
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
            "CURRENT PLAN PROGRESS:\n"
            f"{progress.model_dump_json(indent=2)}\n\n"
            "ACCEPTED CONVERSATION HISTORY:\n"
            f"{self._history_to_text(history)}\n\n"
            "CANDIDATE TUTOR TURN:\n"
            f"{candidate.model_dump_json(indent=2)}"
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=TutorHardCheck,
        )

    @staticmethod
    def _history_to_text(history: list[Message]) -> str:
        messages: list[str] = []

        for index, message in enumerate(history, start=1):
            messages.append(
                f"[{index}] {message['role'].upper()}\n{message['content'].strip()}"
            )

        return "\n\n".join(messages)


class TutorHardCheck(StrictModel):
    technical_error: bool
    learner_state_mismatch: bool
    wrong_active_step: bool
    unjustified_step_completion: bool
    latest_student_turn_not_addressed: bool
    solution_leakage: bool
    malformed_or_incoherent: bool
    serious_repetition: bool

    reasons: list[str] = Field(default_factory=list)
    regeneration_feedback: str | None = None

    @property
    def accepted(self) -> bool:
        return not any(
            (
                self.technical_error,
                self.learner_state_mismatch,
                self.wrong_active_step,
                self.unjustified_step_completion,
                self.latest_student_turn_not_addressed,
                self.solution_leakage,
                self.malformed_or_incoherent,
                self.serious_repetition,
            )
        )
