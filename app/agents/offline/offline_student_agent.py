from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from ...common.config import OFFLINE_STUDENT_AGENT_INSTRUCTIONS
from ...common.models import BenchmarkCase, LearnerState, StrictModel
from ..agent import Agent


class StudentVariant(str, Enum):
    RECEPTIVE = "receptive"
    PERSISTENT = "persistent"
    UNCERTAIN = "uncertain"


# Reuse the same learner-state vocabulary that the Tutor predicts.
# During offline synthesis this is the pre-specified target state for the Student turn.
StudentTurnState = LearnerState


STUDENT_TURN_STATE_INSTRUCTIONS = {
    LearnerState.START: """
START
Give only an opening help-seeking turn: briefly report the visible debugging
difficulty and ask the Tutor for help. Do not solve the bug or submit code.
""".strip(),
    LearnerState.CORRECT: """
CORRECT
Give a materially correct response to the Tutor's current request.

Stay focused on the current objective. Do not deliberately abandon it to solve
later objectives, although a direct correct answer may incidentally demonstrate
something useful later.

If the Tutor asks to implement, revise, apply, run, or verify the repaired
program, submit the complete program in proposed_code.
""".strip(),
    LearnerState.INCORRECT: """
INCORRECT
Give a plausible but materially wrong response to the Tutor's current request.
The central answer, reasoning, prediction, or attempted implementation must
actually be wrong. Do not give the correct answer and merely hedge it.
""".strip(),
    LearnerState.QUESTION: """
QUESTION
Primarily ask one relevant technical clarification about the current objective
instead of answering it. Minimal context is allowed, but do not give the full
answer or submit code.
""".strip(),
    LearnerState.COMPREHENSION: """
COMPREHENSION
Demonstrate correct conceptual understanding of the current point in your own
words. Explain a relevant relationship, cause, rule, or implication rather than
giving only a bare answer.

Stay focused on the current objective. A direct explanation may incidentally
demonstrate something useful for a later objective.

If the current Tutor request requires applying or verifying the repaired
program, proposed_code may contain the complete revised program.
""".strip(),
    LearnerState.CONFUSION: """
CONFUSION
Show genuine inability to understand or proceed with the current point and ask
for simpler guidance, an example, or a smaller step. Do not provide the
solution or submit code.
""".strip(),
    LearnerState.IRRELEVANT: """
IRRELEVANT
Give a brief plausible off-topic response. Do not address the current debugging
objective, provide solution progress, or submit code.
""".strip(),
}


# ScaffoldLM data-synthesis distribution for turns after START:
# Correct, Incorrect, Question, Comprehension, Confusion, Irrelevant.
STUDENT_TURN_STATE_WEIGHTS = {
    LearnerState.CORRECT: 0.50,
    LearnerState.INCORRECT: 0.20,
    LearnerState.QUESTION: 0.10,
    LearnerState.COMPREHENSION: 0.10,
    LearnerState.CONFUSION: 0.05,
    LearnerState.IRRELEVANT: 0.05,
}


STUDENT_VARIANT_INSTRUCTIONS = {
    StudentVariant.RECEPTIVE: """
RECEPTIVE
Use an open, cooperative tone. When the sampled state is CORRECT or
COMPREHENSION, the Student may accept or use Tutor evidence readily. When the
sampled state is INCORRECT, QUESTION, CONFUSION, or IRRELEVANT, realize that
state anyway. This tendency never overrides the sampled learner state.
""".strip(),
    StudentVariant.PERSISTENT: """
PERSISTENT
Use a somewhat resistant or self-defending tone when compatible with the
sampled state. For INCORRECT, QUESTION, or CONFUSION, existing profile beliefs
can be useful content. For CORRECT or COMPREHENSION, still produce the required
correct behaviour. This tendency never overrides the sampled learner state.
""".strip(),
    StudentVariant.UNCERTAIN: """
UNCERTAIN
Use tentative wording such as "I think", "maybe", or "does that mean..." when
compatible with the sampled state. Uncertainty changes confidence and phrasing,
not whether the response is correct, incorrect, questioning, confused, or
irrelevant. The sampled learner state is authoritative.
""".strip(),
}


class StudentProfile(StrictModel):
    beliefs: list[str] = Field(min_length=1, max_length=5)


class StudentTurn(StrictModel):
    reply: str
    proposed_code: str = ""


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

    @staticmethod
    def _build_instructions(
        *,
        base_instructions: str,
        profile: StudentProfile,
        variant: StudentVariant,
    ) -> str:
        return (
            f"{base_instructions}\n\n"
            "# Private student profile for this conversation\n\n"
            "The following profile defines beliefs and assumptions this Student "
            "currently holds. Treat them as genuine beliefs; do not infer that "
            "they are correct or incorrect merely because they appear in the "
            "profile. Use them only to control the Student's reasoning and "
            "behaviour.\n\n"
            "<student_profile>\n"
            f"{profile.model_dump_json(indent=2)}\n"
            "</student_profile>\n\n"
            "# Private behaviour tendency\n\n"
            f"Variant: {variant.value.upper()}\n\n"
            f"{STUDENT_VARIANT_INSTRUCTIONS[variant]}\n\n"
            """The sampled learner state is authoritative for each turn.
            StudentProfile and behaviour tendency are secondary content/style cues only.
            Never change the semantic state of a reply to preserve profile continuity or
            variant behaviour."""
        )

    def generate_turn(
        self,
        *,
        dialogue_history: str,
        is_first_turn: bool,
        turn_state: StudentTurnState,
        regeneration_feedback: str = "",
    ) -> StudentTurn:
        if is_first_turn:
            task = (
                "Generate the first Student turn. Briefly describe the observed "
                "difficulty with the buggy program and ask the Tutor for help. "
                "Begin from the supplied StudentProfile beliefs. Do not provide "
                "corrected code or an exact final repair on this first turn, and do "
                "not fully solve the debugging problem before tutoring has begun."
            )
        else:
            task = (
                "Generate one Student turn that realizes the supplied target learner "
                "state. Use the Tutor's latest message only to identify the current "
                "topic or task; the sampled state controls the semantic behaviour."
            )

        prompt = (
            f"{task}\n\n"
            "<target_learner_state>\n"
            f"State: {turn_state.value.upper()}\n"
            f"{STUDENT_TURN_STATE_INSTRUCTIONS[turn_state]}\n"
            "</target_learner_state>\n\n"
            "<programming_case>\n"
            f"{self.case.visible_context()}\n"
            "</programming_case>\n\n"
            "<conversation_history>\n"
            f"{dialogue_history}\n"
            "</conversation_history>"
        )

        if regeneration_feedback.strip():
            prompt += (
                "\n\n<regeneration_feedback>\n"
                "The previous Student turn was rejected because its behaviour "
                "was not suitable for this simulated conversation. Generate a new "
                "response to the same conversation while correcting this issue:\n"
                f"{regeneration_feedback.strip()}\n"
                "</regeneration_feedback>"
            )

        return self._get_structured_output(
            prompt=prompt,
            output_type=StudentTurn,
        )
