from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from ..common.config import STUDENT_AGENT_INSTRUCTIONS
from ..common.conversation import history_to_text
from ..common.message import Message
from ..common.models import BenchmarkCase, StrictModel
from .agent import Agent


class StudentAgent(Agent):
    """
    Simulates one persistent student for one programming case.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        case: BenchmarkCase,
        profile: StudentProfile,
        instructions: str = STUDENT_AGENT_INSTRUCTIONS,
    ):
        personalized_instructions = self._build_instructions(
            base_instructions=instructions,
            profile=profile,
        )

        super().__init__(
            llm=llm,
            model=model,
            instructions=personalized_instructions,
        )

        self.case = case
        self.profile = profile

    @staticmethod
    def _build_instructions(
        *,
        base_instructions: str,
        profile: StudentProfile,
    ) -> str:
        return (
            f"{base_instructions}\n\n"
            "# Private student profile for this conversation\n\n"
            "The following profile defines the student's persistent "
            "incorrect beliefs. Use it only to control the student's "
            "reasoning and behaviour.\n\n"
            "<student_profile>\n"
            f"{profile.model_dump_json(indent=2)}\n"
            "</student_profile>"
        )

    def generate_turn(
        self,
        history: list[Message],
    ) -> StudentTurn:
        if history:
            task = (
                "Generate the student's next response to the tutor's latest "
                "message. Continue from the existing conversation."
            )
        else:
            task = (
                "Generate the first student turn. Briefly describe the "
                "difficulty with the buggy program and ask the tutor for help."
            )

        prompt = (
            f"{task}\n\n"
            "<programming_case>\n"
            f"{self.case.visible_context(self.case.observed_failure)}\n"
            "</programming_case>\n\n"
            "<conversation_history>\n"
            f"{history_to_text}\n"
            "</conversation_history>"
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=StudentTurn,
        )


class StudentProfile(StrictModel):
    misconceptions: list[str] = Field(min_length=1, max_length=3)


class StudentTurn(StrictModel):
    learner_state: LearnerState
    reply: str
    proposed_code: str

    def to_message(self) -> Message:
        """
        Convert the turn into the visible message shown to the Tutor.
        The hidden learner_state is deliberately excluded.
        """

        content = self.reply.strip()

        if self.proposed_code.strip():
            content += (
                f"\n\nProposed code:\n```python\n{self.proposed_code.rstrip()}\n```"
            )

        return Message(
            role="student",
            content=content,
        )


class LearnerState(str, Enum):
    START = "START"
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    QUESTION = "QUESTION"
    COMPREHENSION = "COMPREHENSION"
    CONFUSION = "CONFUSION"
    IRRELEVANT = "IRRELEVANT"
    END = "END"
