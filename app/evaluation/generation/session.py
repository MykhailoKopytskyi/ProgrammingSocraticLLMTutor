from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agents.offline.offline_student_agent import StudentVariant
from app.common.models import BenchmarkCase

from ..backends import ModelProtocolError
from ..models import EvaluationCodeExecution, EvaluationRound, EvaluationSessionResult
from ..storage import JsonlStore
from .student import StudentGenerationError, StudentStateSampler, StudentSystem
from .tutor import TutorSystem


class EvaluationSession:
    PROTOCOL_VERSION = "2"

    def __init__(
        self,
        *,
        case: BenchmarkCase,
        student: StudentSystem,
        tutor: TutorSystem,
        student_variant: StudentVariant,
        student_state_seed: int = 0,
        max_rounds: int = 14,
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        self.case = case
        self.student = student
        self.tutor = tutor
        self.student_variant = student_variant
        self.max_rounds = max_rounds
        self.state_sampler = StudentStateSampler(student_state_seed)

    @property
    def session_id(self) -> str:
        return (
            f"protocol-{self.PROTOCOL_VERSION}__{self.case.source}__{self.case.case_id}__"
            f"{self.student_variant.value}__seed-{self.state_sampler.seed}__"
            f"student-{self.student.config_id}__tutor-{self.tutor.config_id}__"
            f"max-rounds-{self.max_rounds}"
        )

    def run(self) -> EvaluationSessionResult:
        rounds: list[EvaluationRound] = []
        current_code = self.case.buggy_code
        solved = False
        termination_reason = "max_rounds"

        try:
            self.student.start_case(self.case, self.student_variant)
        except ModelProtocolError as error:
            return self._result(
                rounds=rounds,
                current_code=current_code,
                solved=False,
                termination_reason="student_setup_protocol_error",
                metadata={"error": str(error)},
            )

        try:
            self.tutor.start_case(self.case)
        except ModelProtocolError as error:
            return self._result(
                rounds=rounds,
                current_code=current_code,
                solved=False,
                termination_reason="tutor_setup_protocol_error",
                metadata={"error": str(error)},
            )

        for round_index in range(self.max_rounds):
            history = tuple(rounds)
            target_state = self.state_sampler.state_for_turn(
                case_id=self.case.case_id,
                variant=self.student_variant,
                round_index=round_index,
            )
            try:
                student_turn, execution_result = self.student.generate_turn(
                    history=history,
                    target_state=target_state,
                    is_first_turn=round_index == 0,
                )
            except (ModelProtocolError, StudentGenerationError) as error:
                return self._result(
                    rounds=rounds,
                    current_code=current_code,
                    solved=False,
                    termination_reason="student_output_protocol_error",
                    metadata={**self.tutor.metadata(), "error": str(error)},
                )

            if student_turn.proposed_code.strip():
                current_code = student_turn.proposed_code

            record = EvaluationRound(
                round_index=round_index + 1,
                mastery_before=self.tutor.mastery_state(),
                student=student_turn,
                code_execution=EvaluationCodeExecution.from_result(execution_result),
            )
            try:
                record.tutor = self.tutor.generate_turn(
                    history=history,
                    student_turn=student_turn,
                    latest_code_execution=execution_result,
                    current_working_code=current_code,
                )
                record.mastery_after = self.tutor.mastery_state()
            except ModelProtocolError as error:
                record.error = str(error)
                rounds.append(record)
                termination_reason = "tutor_output_protocol_error"
                break

            rounds.append(record)
            if execution_result is not None and execution_result.passed:
                solved = True
                termination_reason = "tests_passed"
                break

        return self._result(
            rounds=rounds,
            current_code=current_code,
            solved=solved,
            termination_reason=termination_reason,
            metadata=self.tutor.metadata(),
        )

    def _result(
        self,
        *,
        rounds: list[EvaluationRound],
        current_code: str,
        solved: bool,
        termination_reason: str,
        metadata: dict[str, Any],
    ) -> EvaluationSessionResult:
        return EvaluationSessionResult(
            session_id=self.session_id,
            protocol_version=self.PROTOCOL_VERSION,
            case_id=self.case.case_id,
            source=self.case.source,
            student_variant=self.student_variant,
            student_state_seed=self.state_sampler.seed,
            student_system=self.student.system_id,
            student_config_id=self.student.config_id,
            tutor_system=self.tutor.system_id,
            tutor_config_id=self.tutor.config_id,
            max_rounds=self.max_rounds,
            solved=solved,
            termination_reason=termination_reason,
            final_code=current_code,
            tutor_metadata=metadata,
            rounds=rounds,
        )

class EvaluationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.store = JsonlStore(self.path)

    def append(self, result: EvaluationSessionResult) -> None:
        self.store.append_model(result)

    def completed_session_ids(self) -> set[str]:
        result: set[str] = set()
        for row in self.store.read_mappings():
            try:
                session_id = str(row["session_id"])
            except KeyError as error:
                raise ValueError(
                    f"Evaluation record in {self.path} is missing session_id"
                ) from error
            if session_id in result:
                raise ValueError(
                    f"Duplicate evaluation session_id {session_id} in {self.path}"
                )
            result.add(session_id)
        return result
