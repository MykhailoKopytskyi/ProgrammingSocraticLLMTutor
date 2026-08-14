from __future__ import annotations

from ..agents.offline.offline_dialogue_verifier_agent import DialogueVerification
from ..agents.offline.offline_plan_verifier_agent import PlanVerification
from ..agents.offline.offline_student_agent import (
    StudentProfile,
    StudentTurn,
    StudentVariant,
)
from ..agents.offline.offline_student_turn_verifier_agent import StudentTurnCheck
from ..agents.offline.offline_tutor_agent import TutorTurn
from ..agents.offline.offline_tutor_turn_verifier_agent import TutorHardCheck
from ..common.code_runner import TestRunResult
from ..common.models import PlannerOutput, StrictModel


class CodeExecutionRecord(StrictModel):
    passed: bool
    exit_code: int
    output: str
    timed_out: bool

    @classmethod
    def from_result(cls, result: TestRunResult) -> CodeExecutionRecord:
        return cls(
            passed=result.passed,
            exit_code=result.exit_code,
            output=result.output,
            timed_out=result.timed_out,
        )


class StudentTurnRecord(StrictModel):
    turn: StudentTurn
    code_execution: CodeExecutionRecord | None = None
    state_check: StudentTurnCheck


class TutorTurnRecord(StrictModel):
    turn: TutorTurn
    hard_check: TutorHardCheck


class GeneratedDialogue(StrictModel):
    dialogue_id: str
    case_id: str
    source: str
    student_variant: StudentVariant
    student_profile: StudentProfile
    planner_output: PlannerOutput
    plan_verification: PlanVerification
    plan_attempts: int
    turns: list[StudentTurnRecord | TutorTurnRecord]
    dialogue_verification: DialogueVerification
