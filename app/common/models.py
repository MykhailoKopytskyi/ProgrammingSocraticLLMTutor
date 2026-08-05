from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class BugAnnotation(StrictModel):
    bug_id: str
    description: str
    fix: str


class BenchmarkCase(StrictModel):
    case_id: str
    problem_statement: str
    buggy_code: str
    tests: str = ""
    student_question: str
    observed_failure: str = ""

    bugs: list[BugAnnotation] = Field(min_length=1, max_length=3)

    correct_code: str = ""
    student_misconceptions: list[str] = Field(default_factory=list, max_length=3)

    source: str = "manual"

    # Validate the required fields of this object to make sure they are not empty
    @model_validator(mode="after")
    def validate_non_empty_fields(self) -> BenchmarkCase:
        required_text = {
            "case_id": self.case_id,
            "problem_statement": self.problem_statement,
            "buggy_code": self.buggy_code,
            # "tests": self.tests,
            # "correct_code": self.correct_code,
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


class PedagogicalPlan(StrictModel):
    plan_summary: str

    steps: list[PlanStep] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def validate_unique_step_ids(self) -> PedagogicalPlan:
        step_ids = [step.step_id for step in self.steps]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Plan step IDs must be unique")

        return self


class PlannerOutput(StrictModel):
    """
    Complete output of the solve-and-plan task.

    corrected_code is a candidate until execution and oracle verification
    both accept it.
    """

    diagnosis_summary: str
    corrected_code: str
    plan: PedagogicalPlan

    @model_validator(mode="after")
    def validate_content(self) -> PlannerOutput:
        if not self.diagnosis_summary.strip():
            raise ValueError("diagnosis_summary must not be empty")

        if not self.corrected_code.strip():
            raise ValueError("corrected_code must not be empty")

        return self


class PlanProgress(StrictModel):
    completed_step_ids: list[str]
    active_step_id: str


class TutorTurnQualityEvaluation(StrictModel):
    epistemic_soundness: int = Field(ge=0, le=2)
    target_alignment: int = Field(ge=0, le=2)
    disclosure_level: int = Field(ge=0, le=4)
    reasoning_elicitation: int = Field(ge=0, le=2)
    contingency: int = Field(ge=0, le=2)
    notes: str = ""


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
