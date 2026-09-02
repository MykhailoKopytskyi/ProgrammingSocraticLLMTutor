from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from app.agents.offline.offline_student_agent import StudentVariant
from app.common.code_runner import TestRunResult
from app.common.models import LearnerState, StrictModel


class StepStatus(str, Enum):
    DEMONSTRATED = "DEMONSTRATED"
    NOT_DEMONSTRATED = "NOT_DEMONSTRATED"


class RuntimePlanStep(StrictModel):
    step_id: str
    target_concept: str
    guiding_question: str
    expected_answer: str


class RuntimePlan(StrictModel):
    plan_summary: str
    steps: list[RuntimePlanStep] = Field(min_length=3, max_length=8)

    @model_validator(mode="after")
    def validate_steps(self) -> RuntimePlan:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Plan step IDs must be unique")
        expected = [f"step_{index}" for index in range(1, len(step_ids) + 1)]
        if step_ids != expected:
            raise ValueError(f"Plan step IDs must be consecutive: expected {expected}, got {step_ids}")
        return self


class StepAssessment(StrictModel):
    step_id: str
    status: StepStatus


class SocraticRepairTutorOutput(StrictModel):
    learner_state: LearnerState
    step_assessments: list[StepAssessment]
    reply: str = Field(min_length=1)


class BaselineTutorOutput(StrictModel):
    reply: str = Field(min_length=1)


class EvaluationStudentTurn(StrictModel):
    target_learner_state: LearnerState
    reply: str = Field(min_length=1)
    proposed_code: str = ""
    generation_attempts: int = Field(ge=1)


class EvaluationCodeExecution(StrictModel):
    passed: bool
    exit_code: int
    output: str
    timed_out: bool

    @classmethod
    def from_result(cls, result: TestRunResult | None) -> EvaluationCodeExecution | None:
        if result is None:
            return None
        return cls(
            passed=result.passed,
            exit_code=result.exit_code,
            output=result.output,
            timed_out=result.timed_out,
        )


class MasteryState(StrictModel):
    demonstrated_step_ids: list[str] = Field(default_factory=list)
    active_step_id: str | None = None
    undemonstrated_step_ids: list[str] = Field(default_factory=list)


class EvaluationTutorTurn(StrictModel):
    system_id: str
    reply: str = Field(min_length=1)
    raw_output: str = ""
    learner_state: LearnerState | None = None
    step_assessments: list[StepAssessment] = Field(default_factory=list)
    protocol_violations: list[str] = Field(default_factory=list)


class EvaluationRound(StrictModel):
    round_index: int = Field(ge=1)
    mastery_before: MasteryState | None = None
    student: EvaluationStudentTurn
    code_execution: EvaluationCodeExecution | None = None
    tutor: EvaluationTutorTurn | None = None
    mastery_after: MasteryState | None = None
    error: str | None = None


class EvaluationSessionResult(StrictModel):
    session_id: str
    protocol_version: str
    case_id: str
    source: str
    student_variant: StudentVariant
    student_state_seed: int
    student_system: str
    student_config_id: str = ""
    tutor_system: str
    tutor_config_id: str = ""
    max_rounds: int
    solved: bool
    termination_reason: str
    final_code: str
    tutor_metadata: dict[str, Any] = Field(default_factory=dict)
    rounds: list[EvaluationRound] = Field(default_factory=list)
