from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .message import Message


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class LearnerState(str, Enum):
    START = "START"
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    QUESTION = "QUESTION"
    COMPREHENSION = "COMPREHENSION"
    CONFUSION = "CONFUSION"
    IRRELEVANT = "IRRELEVANT"
    END = "END"


class TutorAction(str, Enum):
    ASK = "ASK"
    ADVANCE = "ADVANCE"
    REASK = "REASK"
    HINT = "HINT"
    SIMPLIFY = "SIMPLIFY"
    ANSWER_AND_STEER = "ANSWER_AND_STEER"
    REFOCUS = "REFOCUS"
    SUMMARY = "SUMMARY"


class BugAnnotation(StrictModel):
    bug_id: str
    description: str
    fix: str


class BenchmarkCase(StrictModel):
    case_id: str
    problem_statement: str
    buggy_code: str
    tests: str
    student_question: str
    observed_failure: str = ""

    bugs: list[BugAnnotation] = Field(min_length=1, max_length=3)

    correct_code: str
    student_misconceptions: list[str] = Field(default_factory=list, max_length=3)

    source: str = "manual"

    # Validate the required fields of this object to make sure they are not empty
    @model_validator(mode="after")
    def validate_non_empty_fields(self) -> BenchmarkCase:
        required_text = {
            "case_id": self.case_id,
            "problem_statement": self.problem_statement,
            "buggy_code": self.buggy_code,
            "tests": self.tests,
            "correct_code": self.correct_code,
            "student_question": self.student_question,
        }
        missing = []
        for name, value in required_text.items():
            if not value.strip():
                missing.append(name)

        if len(missing) > 0:
            raise ValueError(f"Empty required fields: {', '.join(missing)}")
        return self

    def visible_context(self, observed_failure: str | None = None) -> str:
        """
        Information available to the Planner and Tutor.
        Excludes annotated bugs, fixes and corrected code
        """

        failure = (
            observed_failure if observed_failure is not None else self.observed_failure
        )
        return (
            f"Problem statement:\n"
            f"{self.problem_statement.strip()}\n\n"
            f"Buggy code:\n"
            f"```python\n"
            f"{self.buggy_code.rstrip()}\n"
            f"```\n\n"
            f"Tests:\n"
            f"```pytest\n"
            f"{self.tests.rstrip()}\n"
            f"```\n\n"
            f"Observed test output:\n"
            f"{failure.strip() or '[not executed]'}\n\n"
            f"Student question/confusion:\n"
            f"{self.student_question.strip()}"
        )

    def oracle_context(self) -> str:
        "Training only information available to verification agents"

        bug_text = "\n".join(
            (f"- {bug.bug_id}: {bug.description}\n  Required fix: {bug.fix}")
            for bug in self.bugs
        )

        correct = self.correct_code.strip() or "[not supplied]"

        return (
            f"Ground-truth bugs and fixes:\n"
            f"{bug_text}\n\n"
            f"Corrected code:\n"
            f"```python\n"
            f"{correct}\n"
            f"```"
        )


class PlanStep(StrictModel):
    step_id: str
    target_concept: str
    guiding_question: str
    expected_answer: str

    related_bug_ids: list[str]
    prerequisite_step_ids: list[str]

    max_disclosure_level: int = Field(ge=0, le=4)


class PedagogicalPlan(StrictModel):
    plan_summary: str

    steps: list[PlanStep] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def validate_ids_and_order(self) -> PedagogicalPlan:
        step_ids = []
        for step in self.steps:
            step_ids.append(step.step_id)

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Plan step IDs must be unique")

        known_step_ids: set[str] = set()

        for step in self.steps:
            unknown_prerequisites = set(step.prerequisite_step_ids) - known_step_ids

            if unknown_prerequisites:
                raise ValueError(
                    f"Step {step.step_id} references future or unknown"
                    f"prerequisites: {sorted(unknown_prerequisites)}"
                )
            known_step_ids.add(step.step_id)

        return self


class PlanProgress(StrictModel):
    completed_step_ids: list[str]
    active_step_id: str


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


class TutorTurn(StrictModel):
    analysis_and_decision: str
    learner_state: LearnerState
    active_step_id: str
    step_completed: bool
    tutor_action: TutorAction
    reply: str


class PlanVerification(StrictModel):
    accepted: bool

    covered_bug_ids: list[str]
    missing_bug_ids: list[str]
    invented_or_unsupported_claims: list[str]
    errors: list[str]

    regeneration_feedback: str


class TutorEvaluation(StrictModel):
    accepted: bool
    hard_failure_reasons: list[str]

    epistemic_soundness: int = Field(ge=0, le=2)
    target_alignment: int = Field(ge=0, le=2)
    disclosure_level: int = Field(ge=0, le=4)
    reasoning_elicitation: int = Field(ge=0, le=2)
    contingency: int = Field(ge=0, le=2)

    learner_state_label_correct: bool  # true if the tutor's label matches the student message and false otherwise
    active_step_correct: (
        bool  # true if the tutor discusses the coorect current plan step
    )
    serious_repetition: (
        bool  # true if the tutor is repeating itself without adding useful guidance
    )

    evidence: str  # a short justification for evaluation
    regeneration_feedback: str | None = (
        None  # instructions given to the Tutor when the response is rejected
    )


# If the dataset misses tests then we can automatically generate them
class GeneratedTest(StrictModel):
    test_id: str
    test_code: str
    purpose: str
    related_bug_ids: list[str]


class GeneratedTests(StrictModel):
    imports_and_fixtures: str = ""
    tests: list[GeneratedTest] = Field(min_length=1)


# If the dataset misses corrected code + its explanation, then we can automatically generate them
class AppliedFix(StrictModel):
    bug_id: str
    explanation: str


class ReferenceRepair(StrictModel):
    corrected_code: str
    applied_fixes: list[AppliedFix]
