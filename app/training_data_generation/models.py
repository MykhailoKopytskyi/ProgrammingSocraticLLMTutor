from __future__ import annotations

from ..agents.offline.offline_dialogue_verifier_agent import DialogueVerification
from ..agents.offline.offline_turn_verifier_agent import TutorHardCheck
from ..agents.student_agent import StudentProfile, StudentTurn
from ..agents.tutor_agent import TutorTurn
from ..common.code_runner import TestRunResult
from ..common.models import PlannerOutput, StrictModel


class PreparedDialogueCase(StrictModel):
    case_id: str
    source: str
    student_profile: StudentProfile
    planner_output: PlannerOutput


class CodeExecutionRecord(StrictModel):
    passed: bool
    output: str

    @classmethod
    def from_result(cls, result: TestRunResult) -> CodeExecutionRecord:
        return cls(passed=result.passed, output=result.output)


class StudentTurnRecord(StrictModel):
    turn: StudentTurn
    code_execution: CodeExecutionRecord | None = None


class TutorTurnRecord(StrictModel):
    turn: TutorTurn
    hard_check: TutorHardCheck


class GeneratedDialogue(StrictModel):
    case_id: str
    source: str
    student_profile: StudentProfile
    planner_output: PlannerOutput
    turns: list[StudentTurnRecord | TutorTurnRecord]
    dialogue_verification: DialogueVerification
