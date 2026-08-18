from __future__ import annotations

from pydantic import Field

from app.agents.offline.offline_plan_verifier_agent import PlanVerification

from ..agents.offline.offline_dialogue_verifier_agent import DialogueVerification
from ..agents.offline.offline_student_agent import (
    StudentProfile,
    StudentTurn,
    StudentTurnState,
    StudentVariant,
)
from ..agents.offline.offline_student_turn_verifier_agent import StudentTurnCheck
from ..agents.offline.offline_tutor_agent import TutorTurn
from ..agents.offline.offline_tutor_turn_verifier_agent import TutorHardCheck
from ..common.code_runner import TestRunResult
from ..common.models import PlannerOutput, StrictModel


class VerifiedPlan(StrictModel):
    output: PlannerOutput
    verification: PlanVerification
    attempts: int


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
    student_state: StudentTurnState | None = None
    code_execution: CodeExecutionRecord | None = None
    hard_check: StudentTurnCheck


class TutorTurnRecord(StrictModel):
    turn: TutorTurn
    hard_check: TutorHardCheck


class DialogueRoundRecord(StrictModel):
    student: StudentTurnRecord
    tutor: TutorTurnRecord | None = None


class DialogueRecords(StrictModel):
    rounds: list[DialogueRoundRecord] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.rounds

    @property
    def pending_student(self) -> StudentTurnRecord | None:
        if not self.rounds:
            return None

        current_round = self.rounds[-1]
        if current_round.tutor is not None:
            return None
        return current_round.student

    def add_student(self, record: StudentTurnRecord) -> None:
        if self.rounds and self.rounds[-1].tutor is None:
            raise ValueError(
                "Cannot add another Student turn before the Tutor responds."
            )

        self.rounds.append(DialogueRoundRecord(student=record))

    def add_tutor(self, record: TutorTurnRecord) -> None:
        if not self.rounds:
            raise ValueError("Cannot add a Tutor turn before a Student turn.")

        current_round = self.rounds[-1]
        if current_round.tutor is not None:
            raise ValueError("The current Student turn already has a Tutor response.")

        current_round.tutor = record

    @staticmethod
    def _student_text(record: StudentTurnRecord) -> str:
        content = record.turn.reply.strip()

        if record.turn.proposed_code.strip():
            content += (
                "\n\nProposed code:\n```python\n"
                f"{record.turn.proposed_code.rstrip()}\n```"
            )

        return content

    def to_student_text(self) -> str:
        if not self.rounds:
            return "[empty]"

        parts: list[str] = []
        index = 1

        for round_record in self.rounds:
            parts.append(
                f"[{index}] STUDENT\n{self._student_text(round_record.student)}"
            )
            index += 1

            if round_record.tutor is not None:
                parts.append(
                    f"[{index}] TUTOR\n{round_record.tutor.turn.reply.strip()}"
                )
                index += 1

        return "\n\n".join(parts)

    def to_tutor_text(self) -> str:
        if not self.rounds:
            return "[empty]"

        parts: list[str] = []
        index = 1

        for round_record in self.rounds:
            content = self._student_text(round_record.student)

            if round_record.tutor is not None:
                content = (
                    f"Tutor-assessed learner state: "
                    f"{round_record.tutor.turn.learner_state.value}\n"
                    f"{content}"
                )

            parts.append(f"[{index}] STUDENT\n{content}")
            index += 1

            if round_record.tutor is not None:
                parts.append(
                    f"[{index}] TUTOR\n{round_record.tutor.turn.reply.strip()}"
                )
                index += 1

        return "\n\n".join(parts)

    def to_verifier_text(self) -> str:
        if not self.rounds:
            return "[empty]"

        parts: list[str] = []

        for round_index, round_record in enumerate(self.rounds, start=1):
            target_state = (
                round_record.student.student_state.value
                if round_record.student.student_state is not None
                else "[not recorded]"
            )
            state_consistent = round_record.student.hard_check.state_consistent
            tutor_assessed_state = (
                round_record.tutor.turn.learner_state.value
                if round_record.tutor is not None
                else "[no Tutor assessment]"
            )
            parts.append(
                f"ROUND {round_index}\n"
                f"Target learner state: {target_state}\n"
                f"Student state consistent with target: {state_consistent}\n"
                f"Tutor-assessed learner state: {tutor_assessed_state}\n"
                f"STUDENT: {self._student_text(round_record.student)}"
            )

            if round_record.tutor is not None:
                parts.append(
                    f"TUTOR: {round_record.tutor.turn.reply.strip()}"
                )

        return "\n\n".join(parts)


class GeneratedDialogue(StrictModel):
    dialogue_id: str
    case_id: str
    source: str
    student_variant: StudentVariant
    student_profile: StudentProfile
    verified_plan: VerifiedPlan
    records: DialogueRecords
    dialogue_verification: DialogueVerification
