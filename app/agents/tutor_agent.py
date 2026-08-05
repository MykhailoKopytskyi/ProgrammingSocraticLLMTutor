from __future__ import annotations

from enum import Enum
from typing import Any

from ..common.config import TUTOR_AGENT_INSTRUCTIONS
from ..common.message import Message
from ..common.models import (
    BenchmarkCase,
    PedagogicalPlan,
    PlanProgress,
    StrictModel,
)
from .agent import Agent
from .student_agent import LearnerState


class TutorAgent(Agent):
    """
    Simulates one tutor for one programming case and one fixed plan.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        instructions: str = TUTOR_AGENT_INSTRUCTIONS,
    ):
        personalized_instructions = self._build_instructions(
            base_instructions=instructions,
            plan=plan,
        )

        super().__init__(
            llm=llm,
            model=model,
            instructions=personalized_instructions,
        )

        self.case = case
        self.plan = plan

    @staticmethod
    def _build_instructions(
        *,
        base_instructions: str,
        plan: PedagogicalPlan,
    ) -> str:
        return (
            f"{base_instructions}\n\n"
            "# Private pedagogical plan for this conversation\n\n"
            "The following plan is authoritative private tutoring guidance. "
            "Follow it throughout the conversation. Do not mention the plan, "
            "its expected answers, bug identifiers, or disclosure levels to "
            "the student.\n\n"
            "<pedagogical_plan>\n"
            f"{plan.model_dump_json(indent=2)}\n"
            "</pedagogical_plan>"
        )

    def generate_turn(
        self,
        *,
        history: list[Message],
        progress: PlanProgress,
        regeneration_feedback: str = "",
    ) -> TutorTurn:
        if not history:
            raise ValueError("TutorAgent requires a student message in the history")

        if history[-1]["role"] != "student":
            raise ValueError(
                "Conversation history must end with the latest student message"
            )

        prompt = (
            "Generate the tutor's next conversational turn in response to "
            "the final student message in the conversation history.\n\n"
            "<programming_case>\n"
            f"{self.case.visible_context(self.case.observed_failure)}\n"
            "</programming_case>\n\n"
            "<plan_progress>\n"
            f"{progress.model_dump_json(indent=2)}\n"
            "</plan_progress>\n\n"
            "<conversation_history>\n"
            f"{self.history_to_text(history)}\n"
            "</conversation_history>"
        )

        if regeneration_feedback.strip():
            prompt += (
                "\n\n<regeneration_feedback>\n"
                "The previous tutor candidate was rejected. Correct these "
                "problems while responding to the same final student message:\n"
                f"{regeneration_feedback.strip()}\n"
                "</regeneration_feedback>"
            )

        return self._get_structured_output(
            prompt=prompt,
            output_type=TutorTurn,
        )

    def history_to_text(history: list[Message]) -> str:
        if not history:
            return "[empty]"

        messages: list[str] = []

        for index, message in enumerate(history, start=1):
            messages.append(
                f"[{index}] {message['role'].upper()}\n{message['content'].strip()}"
            )

        return "\n\n".join(messages)


class TutorTurn(StrictModel):
    analysis_and_decision: str
    learner_state: LearnerState
    active_step_id: str
    step_completed: bool
    tutor_action: TutorAction
    reply: str


class TutorAction(str, Enum):
    ASK = "ASK"
    ADVANCE = "ADVANCE"
    REASK = "REASK"
    HINT = "HINT"
    SIMPLIFY = "SIMPLIFY"
    ANSWER_AND_STEER = "ANSWER_AND_STEER"
    REFOCUS = "REFOCUS"
    SUMMARY = "SUMMARY"
