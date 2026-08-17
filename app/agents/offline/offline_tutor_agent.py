from __future__ import annotations

from enum import Enum
from typing import Any

from ...common.code_runner import TestRunResult
from ...common.config import OFFLINE_TUTOR_AGENT_INSTRUCTIONS
from ...common.models import (
    BenchmarkCase,
    LearnerState,
    PlannerOutput,
    PlanProgress,
    StrictModel,
)
from ..agent import Agent


class TutorTurn(StrictModel):
    analysis_and_decision: str
    completed_through_step_id: str | None
    tutor_action: TutorAction
    reply: str
    learner_state: LearnerState


class TutorAction(str, Enum):
    ASK = "ASK"
    ADVANCE = "ADVANCE"
    REASK = "REASK"
    HINT = "HINT"
    SIMPLIFY = "SIMPLIFY"
    ANSWER_AND_STEER = "ANSWER_AND_STEER"
    REFOCUS = "REFOCUS"
    SUMMARY = "SUMMARY"


class OfflineTutorAgent(Agent):
    """Simulates one tutor for one programming case and one fixed plan."""

    def __init__(
        self,
        llm: Any,
        model: str,
        case: BenchmarkCase,
        planner_output: PlannerOutput,
        instructions: str = OFFLINE_TUTOR_AGENT_INSTRUCTIONS,
    ):
        personalized_instructions = self._build_instructions(
            base_instructions=instructions,
            planner_output=planner_output,
        )

        super().__init__(
            llm=llm,
            model=model,
            instructions=personalized_instructions,
            max_output_tokens=2500,
        )

        self.case = case

    @staticmethod
    def _build_instructions(
        *,
        base_instructions: str,
        planner_output: PlannerOutput,
    ) -> str:
        return (
            f"{base_instructions}\n\n"
            "# Private planner output for this conversation\n\n"
            "The following planner output is authoritative private tutoring "
            "grounding. It contains the diagnosed bugs, their required fixes, "
            "a corrected solution, and the pedagogical plan. Use it to keep "
            "your tutoring technically grounded, but do not disclose this "
            "private information prematurely.\n\n"
            "<planner_output>\n"
            f"{planner_output.model_dump_json(indent=2)}\n"
            "</planner_output>"
        )

    def generate_turn(
        self,
        *,
        verified_history: str,
        progress: PlanProgress,
        latest_code_execution: TestRunResult | None = None,
        regeneration_feedback: str = "",
    ) -> TutorTurn:
        execution_evidence = (
            "No student-submitted code has been executed yet."
            if latest_code_execution is None
            else (
                f"passed={latest_code_execution.passed}\n{latest_code_execution.output}"
            )
        )

        prompt = (
            "Generate the tutor's next conversational turn in response to "
            "the final student message in the conversation history.\n\n"
            "<programming_case>\n"
            f"{self.case.visible_context()}\n"
            "</programming_case>\n\n"
            "<plan_progress>\n"
            f"{progress.model_dump_json(indent=2)}\n"
            "</plan_progress>\n\n"
            "<latest_student_code_execution>\n"
            f"{execution_evidence}\n"
            "</latest_student_code_execution>\n\n"
            "<conversation_history>\n"
            f"{verified_history}\n"
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
