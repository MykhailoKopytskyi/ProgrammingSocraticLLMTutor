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
    tests: str
    observed_failure: str
    bugs: list[BugAnnotation] = Field(min_length=1, max_length=3)
    correct_code: str
    source: str = "manual"

    @model_validator(mode="after")
    def validate_non_empty_fields(self) -> BenchmarkCase:
        required_text = {
            "case_id": self.case_id,
            "problem_statement": self.problem_statement,
            "buggy_code": self.buggy_code,
            "tests": self.tests,
            "observed_failure": self.observed_failure,
            "correct_code": self.correct_code,
        }
        missing = []
        for name, value in required_text.items():
            if not value.strip():
                missing.append(name)

        if missing:
            raise ValueError(f"Empty required fields: {', '.join(missing)}")
        return self

    def visible_context(self) -> str:
        """Return the case information that is visible at runtime."""
        return (
            f"Problem statement:\n{self.problem_statement.strip()}\n\n"
            f"Buggy code:\n```python\n{self.buggy_code.rstrip()}\n```\n\n"
            f"Tests:\n```pytest\n{self.tests.rstrip()}\n```\n\n"
            f"Observed test output:\n"
            f"{self.observed_failure.strip() or '[not executed]'}\n\n"
        )

    def oracle_context(self) -> str:
        """Return training-only bug and reference-solution information."""
        bug_lines = []
        for bug in self.bugs:
            bug_lines.append(
                f"- {bug.bug_id}: {bug.description}\n  Required fix: {bug.fix}"
            )

        bug_text = "\n".join(bug_lines)
        correct = self.correct_code.strip() or "[not supplied]"
        return (
            f"Ground-truth bugs and fixes:\n{bug_text}\n\n"
            f"Corrected code:\n```python\n{correct}\n```"
        )


class PlanStep(StrictModel):
    step_id: str
    target_concept: str
    guiding_question: str
    expected_answer: str
    related_bug_ids: list[str]


class PedagogicalPlan(StrictModel):
    plan_summary: str
    steps: list[PlanStep] = Field(min_length=2, max_length=7)

    @model_validator(mode="after")
    def validate_unique_step_ids(self) -> PedagogicalPlan:
        step_ids = []
        for step in self.steps:
            step_ids.append(step.step_id)

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Plan step IDs must be unique")
        return self


class PlannerOutput(StrictModel):
    """Planner target used during training and later at runtime."""

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


class GeneratedTest(StrictModel):
    test_id: str
    test_code: str
    purpose: str
    related_bug_ids: list[str]


class GeneratedTests(StrictModel):
    imports_and_fixtures: str = ""
    tests: list[GeneratedTest] = Field(min_length=1)

    def to_pytest_file(self) -> str:
        sections = []
        if self.imports_and_fixtures.strip():
            sections.append(self.imports_and_fixtures.strip())
        for test in self.tests:
            sections.append(test.test_code.strip())
        return "\n\n".join(sections).rstrip() + "\n"


class AppliedFix(StrictModel):
    bug_id: str
    explanation: str


class ReferenceRepair(StrictModel):
    corrected_code: str
    applied_fixes: list[AppliedFix]

    @model_validator(mode="after")
    def validate_repair(self) -> ReferenceRepair:
        if not self.corrected_code.strip():
            raise ValueError("corrected_code must not be empty")

        bug_ids = []
        for applied_fix in self.applied_fixes:
            bug_ids.append(applied_fix.bug_id)
        if len(bug_ids) != len(set(bug_ids)):
            raise ValueError("applied_fixes must use unique bug IDs")
        return self
