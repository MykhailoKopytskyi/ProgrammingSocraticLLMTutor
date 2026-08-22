from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from ...common.config import OFFLINE_STUDENT_AGENT_INSTRUCTIONS
from ...common.models import BenchmarkCase, LearnerState, PlannerOutput, StrictModel
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
Give an opening help-seeking turn based on the visible failure.

You may report the observed exception, failing test, or incorrect output.
Do not diagnose the underlying cause, compare suspicious identifiers,
explain why the failure occurs, propose a repair, or state what code
should be changed.

The opening turn should establish the problem, not solve the first
debugging objective.
For example, instead of:

“num1 is used before it is assigned, so I think the order is wrong.”
we should say:
“I'm getting NameError: num1 is not defined on the first line. I'm not sure why. Can you help me trace it?”
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


STUDENT_TURN_STATE_WEIGHTS_BY_VARIANT = {
    StudentVariant.RECEPTIVE: {
        LearnerState.CORRECT: 0.40,
        LearnerState.INCORRECT: 0.20,
        LearnerState.QUESTION: 0.10,
        LearnerState.COMPREHENSION: 0.10,
        LearnerState.CONFUSION: 0.10,
        LearnerState.IRRELEVANT: 0.10,
    },
    StudentVariant.UNCERTAIN: {
        LearnerState.CORRECT: 0.30,
        LearnerState.INCORRECT: 0.15,
        LearnerState.QUESTION: 0.25,
        LearnerState.COMPREHENSION: 0.10,
        LearnerState.CONFUSION: 0.15,
        LearnerState.IRRELEVANT: 0.05,
    },
    StudentVariant.PERSISTENT: {
        LearnerState.CORRECT: 0.25,
        LearnerState.INCORRECT: 0.40,
        LearnerState.QUESTION: 0.15,
        LearnerState.COMPREHENSION: 0.05,
        LearnerState.CONFUSION: 0.10,
        LearnerState.IRRELEVANT: 0.05,
    },
}

STUDENT_VARIANT_INSTRUCTIONS = {
    StudentVariant.RECEPTIVE: """
RECEPTIVE
Act like a cooperative beginner who is willing to engage with the Tutor's
questions and update their reasoning when the evidence makes sense.
When the sampled state is CORRECT or COMPREHENSION, the Student may readily
accept and apply useful Tutor guidance.
When the sampled state is INCORRECT, QUESTION, CONFUSION, or IRRELEVANT,
realize that state anyway. Do not become artificially agreeable merely because
this Student is receptive.
The sampled learner state always overrides this behavioural tendency.
""".strip(),
    StudentVariant.PERSISTENT: """
PERSISTENT
Act like a misconception-persistent beginner. Treat the beliefs in the private
StudentProfile as beliefs the Student genuinely holds. Do not abandon a belief
merely because the Tutor disagrees or states that something is wrong.
For INCORRECT, reason consistently from a relevant profile belief when
possible.
For QUESTION, question or probe the part of the Tutor's explanation that
conflicts with the Student's current belief.
For CONFUSION, express the conflict between what the Student currently believes
and the evidence or explanation supplied by the Tutor.
When the sampled state is CORRECT or COMPREHENSION, the Student must still
produce the required correct behaviour. When natural, make clear what evidence
or reasoning caused the Student to revise their earlier belief.

The sampled learner state always overrides this behavioural tendency.
""".strip(),
    StudentVariant.UNCERTAIN: """
UNCERTAIN
Act like a low-confidence beginner who often seeks confirmation before
committing to an interpretation.
Use tentative language naturally, but do not add empty hedging to every
sentence.
For QUESTION, ask a precise clarification about the current debugging point.
For CONFUSION, state specifically what is unclear and ask for a smaller or
simpler step.
For CORRECT and COMPREHENSION, the content must still be technically correct,
although it may be expressed with low confidence.
For INCORRECT, the response must still be materially wrong rather than a
correct answer disguised with words such as "maybe".
The sampled learner state always overrides this behavioural tendency.
""".strip(),
}


class StudentProfile(StrictModel):
    beliefs: list[str] = Field(min_length=1, max_length=5)
    education_level: str = "first-year undergraduate student"
    age: int = Field(default=17, ge=14, le=25)

    programming_experience: str = (
        "beginner Python programmer with roughly one semester of experience"
    )


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
        planner_output: PlannerOutput,
        instructions: str = OFFLINE_STUDENT_AGENT_INSTRUCTIONS,
    ):
        personalized_instructions = self._build_instructions(
            base_instructions=instructions,
            profile=profile,
            variant=variant,
            planner_output=planner_output,
        )

        super().__init__(
            llm=llm,
            model=model,
            instructions=personalized_instructions,
            reasoning_effort="low",
            max_output_tokens=2000,
        )
        self.case = case

    @staticmethod
    def _build_instructions(
        *,
        base_instructions: str,
        profile: StudentProfile,
        variant: StudentVariant,
        planner_output: PlannerOutput,
    ) -> str:
        return (
            "# Private reference information\n\n"
            "The following information is training-only grounding for controlled Student "
            "simulation. It contains the correct diagnosis, corrected code, pedagogical "
            "plan, and expected answers. Use it internally to realize the sampled learner "
            "state accurately. Never mention that this private information was supplied, "
            "and do not expose future answers unless the current Tutor request and sampled "
            "learner state require them.\n\n"
            "For CORRECT and COMPREHENSION, use this reference information to make the "
            "response technically correct for the Tutor's current request. For INCORRECT, "
            "use it to construct a plausible materially wrong response rather than "
            "accidentally giving the correct answer.\n\n"
            "<reference_information>\n"
            f"{planner_output.model_dump_json(indent=2)}\n"
            "</reference_information>\n\n"
            f"{base_instructions}\n\n"
            "# Private student profile for this conversation\n\n"
            "The following profile defines the Student's stable learner background "
            "and the beliefs or assumptions they currently hold. Use the background "
            "fields to keep vocabulary, confidence, and assumed programming knowledge "
            "appropriate for this learner. Treat the beliefs as genuine beliefs; do "
            "not infer that they are correct or incorrect merely because they appear "
            "in the profile. Use the beliefs to ground the Student's reasoning when "
            "compatible with the sampled learner state.\n\n"
            "<student_profile>\n"
            f"{profile.model_dump_json(indent=2)}\n"
            "</student_profile>\n\n"
            "# Private behaviour tendency\n\n"
            f"Variant: {variant.value.upper()}\n\n"
            f"{STUDENT_VARIANT_INSTRUCTIONS[variant]}\n\n"
            "The sampled learner state is authoritative for every turn. "
            "StudentProfile and behaviour tendency influence the Student's reasoning, "
            "background, confidence, and style, but never override the semantic "
            "requirements of the sampled learner state."
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
                "Do not provide "
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
