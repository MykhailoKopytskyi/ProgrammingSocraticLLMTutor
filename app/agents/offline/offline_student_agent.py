from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from ...common.config import OFFLINE_STUDENT_AGENT_INSTRUCTIONS
from ...common.conversation import Conversation
from ...common.message import Message
from ...common.models import BenchmarkCase, StrictModel
from ..agent import Agent


class StudentVariant(str, Enum):
    RECEPTIVE = "receptive"
    PERSISTENT = "persistent"
    UNCERTAIN = "uncertain"


STUDENT_VARIANT_INSTRUCTIONS = {
    StudentVariant.RECEPTIVE: """
Engage readily with useful questions and hints. Make reasonable progress when
the Tutor gives enough evidence or asks a productive question. Do not jump
ahead to conclusions that the conversation has not supported.
""".strip(),
    StudentVariant.PERSISTENT: """
Trust your existing reasoning unless the Tutor gives you convincing evidence
or reasoning to reconsider it. Explain and defend your current reasoning when
appropriate. Update your view when the conversation genuinely gives you enough
reason to do so.
""".strip(),
    StudentVariant.UNCERTAIN: """
Be less confident in your reasoning. Ask for clarification when a question or
concept is genuinely unclear and benefit from concrete tracing or narrower
questions. Do not manufacture confusion after the idea has become clear.
""".strip(),
}


class StudentProfile(StrictModel):
    beliefs: list[str] = Field(min_length=1, max_length=5)


class StudentTurn(StrictModel):
    reply: str
    proposed_code: str = ""

    def to_message(self) -> Message:
        """Convert the turn into the visible message shown to the Tutor."""

        content = self.reply.strip()

        if self.proposed_code.strip():
            content += (
                f"\n\nProposed code:\n```python\n{self.proposed_code.rstrip()}\n```"
            )

        return Message(
            role="student",
            content=content,
        )


class OfflineStudentAgent(Agent):
    """Simulates one persistent student for one programming case."""

    def __init__(
        self,
        llm: Any,
        model: str,
        case: BenchmarkCase,
        profile: StudentProfile,
        variant: StudentVariant,
        instructions: str = OFFLINE_STUDENT_AGENT_INSTRUCTIONS,
    ):
        personalized_instructions = self._build_instructions(
            base_instructions=instructions, profile=profile, variant=variant
        )

        super().__init__(
            llm=llm,
            model=model,
            instructions=personalized_instructions,
        )

        self.case = case
        self.profile = profile
        self.variant = variant

    @staticmethod
    def _build_instructions(
        *,
        base_instructions: str,
        profile: StudentProfile,
        variant: StudentVariant,
    ) -> str:
        return f"{base_instructions}\n\n# Private student profile for this conversation\n\n The following profile defines the student's persistent incorrect beliefs. Use it only to control the student's reasoning and behaviour.\n\n <student_profile>\n{profile.model_dump_json(indent=2)}\n</student_profile>\n\n# Private behaviour tendency\n\n{STUDENT_VARIANT_INSTRUCTIONS[variant]}"

    def generate_turn(
        self,
        conversation: Conversation,
        regeneration_feedback: str = "",
    ) -> StudentTurn:
        if conversation.is_empty:
            task = (
                "Generate the first Student turn. Briefly describe the observed "
                "difficulty with the buggy program and ask the Tutor for help."
            )
        else:
            task = (
                "Respond naturally to the Tutor's latest message given your current "
                "beliefs, the conversation so far, and your behaviour tendency."
            )

        prompt = (
            f"{task}\n\n"
            "<programming_case>\n"
            f"{self.case.visible_context()}\n"
            "</programming_case>\n\n"
            "<conversation_history>\n"
            f"{conversation.to_text()}\n"
            "</conversation_history>"
        )

        if regeneration_feedback.strip():
            prompt += (
                "\n\n<regeneration_feedback>\n"
                "The previous Student turn was rejected because its private state "
                "or behaviour was inconsistent. Generate a new response to the "
                "same conversation while correcting this issue:\n"
                f"{regeneration_feedback.strip()}\n"
                "</regeneration_feedback>"
            )

        return self._get_structured_output(
            prompt=prompt,
            output_type=StudentTurn,
        )
