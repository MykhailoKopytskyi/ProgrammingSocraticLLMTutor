from __future__ import annotations

from typing import Any

from ..agents.offline.offline_dialogue_verifier_agent import (
    OfflineDialogueVerifierAgent,
)
from ..agents.offline.offline_student_agent import (
    LearnerState,
    OfflineStudentAgent,
    StudentProfile,
    StudentTurn,
    StudentVariant,
)
from ..agents.offline.offline_student_turn_verifier_agent import (
    OfflineStudentTurnVerifierAgent,
    StudentTurnCheck,
)
from ..agents.offline.offline_tutor_agent import OfflineTutorAgent, TutorTurn
from ..agents.offline.offline_tutor_turn_verifier_agent import (
    OfflineTutorTurnVerifierAgent,
    TutorHardCheck,
)
from ..common.code_runner import CodeRunner, TestRunResult
from ..common.conversation import Conversation
from ..common.models import BenchmarkCase, PedagogicalPlan, PlanProgress
from .models import (
    CodeExecutionRecord,
    GeneratedDialogue,
    StudentTurnRecord,
    TutorTurnRecord,
)
from .plan_generation import VerifiedPlan


class DialogueGenerationError(RuntimeError):
    pass


class DialogueGenerator:
    def __init__(
        self,
        *,
        llm: Any,
        student_model: str,
        tutor_model: str,
        student_turn_verifier: OfflineStudentTurnVerifierAgent,
        tutor_turn_verifier: OfflineTutorTurnVerifierAgent,
        dialogue_verifier: OfflineDialogueVerifierAgent,
        code_runner: CodeRunner,
        max_rounds: int = 12,
        max_student_attempts: int = 3,
        max_tutor_attempts: int = 3,
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        if max_student_attempts < 1:
            raise ValueError("max_student_attempts must be at least 1")

        if max_tutor_attempts < 1:
            raise ValueError("max_tutor_attempts must be at least 1")

        self._llm = llm
        self._student_model = student_model
        self._tutor_model = tutor_model
        self._student_turn_verifier = student_turn_verifier
        self._tutor_turn_verifier = tutor_turn_verifier
        self._dialogue_verifier = dialogue_verifier
        self._code_runner = code_runner
        self._max_rounds = max_rounds
        self._max_student_attempts = max_student_attempts
        self._max_tutor_attempts = max_tutor_attempts

    def generate(
        self,
        *,
        case: BenchmarkCase,
        verified_plan: VerifiedPlan,
        profile: StudentProfile,
        variant: StudentVariant,
    ) -> GeneratedDialogue:
        dialogue_id = f"{case.case_id}__{variant.value}"
        plan = verified_plan.output.plan

        student = OfflineStudentAgent(
            llm=self._llm,
            model=self._student_model,
            case=case,
            profile=profile,
            variant=variant,
        )

        tutor = OfflineTutorAgent(
            llm=self._llm,
            model=self._tutor_model,
            case=case,
            planner_output=verified_plan.output,
        )

        progress = PlanProgress(
            completed_step_ids=[],
            active_step_id=plan.steps[0].step_id,
        )

        conversation = Conversation()
        records: list[StudentTurnRecord | TutorTurnRecord] = []
        latest_code_execution: TestRunResult | None = None

        for _ in range(self._max_rounds):
            student_turn, student_execution, student_check = (
                self._generate_verified_student_turn(
                    case=case,
                    plan=plan,
                    progress=progress,
                    profile=profile,
                    conversation=conversation,
                    student=student,
                )
            )
            conversation.add(student_turn.to_message())

            if student_execution is not None:
                latest_code_execution = student_execution

            records.append(
                StudentTurnRecord(
                    turn=student_turn,
                    code_execution=(
                        CodeExecutionRecord.from_result(student_execution)
                        if student_execution is not None
                        else None
                    ),
                    state_check=student_check,
                )
            )

            tutor_turn, hard_check = self._generate_verified_tutor_turn(
                case=case,
                plan=plan,
                progress=progress,
                conversation=conversation,
                tutor=tutor,
                expected_learner_state=student_check.learner_state,
                latest_code_execution=latest_code_execution,
            )

            conversation.add(tutor_turn.to_message())
            records.append(
                TutorTurnRecord(
                    turn=tutor_turn,
                    hard_check=hard_check,
                )
            )
            progress = self._update_progress(
                plan=plan,
                progress=progress,
                tutor_turn=tutor_turn,
            )

            if self._is_complete(
                plan=plan,
                progress=progress,
                tutor_turn=tutor_turn,
                latest_code_execution=latest_code_execution,
            ):
                if latest_code_execution is None:
                    raise DialogueGenerationError(
                        "Dialogue completed without code execution."
                    )

                return self._finalise(
                    dialogue_id=dialogue_id,
                    case=case,
                    verified_plan=verified_plan,
                    profile=profile,
                    variant=variant,
                    conversation=conversation,
                    records=records,
                    progress=progress,
                    latest_code_execution=latest_code_execution,
                )

        raise DialogueGenerationError(
            "Conversation reached the maximum number of rounds without valid completion."
        )

    def _generate_verified_student_turn(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        profile: StudentProfile,
        conversation: Conversation,
        student: OfflineStudentAgent,
    ) -> tuple[StudentTurn, TestRunResult | None, StudentTurnCheck]:
        feedback = ""

        for _ in range(self._max_student_attempts):
            candidate = student.generate_turn(
                conversation=conversation,
                regeneration_feedback=feedback,
            )

            # deterministic_error = self._student_candidate_error(
            #     candidate=candidate,
            #     conversation=conversation,
            # )

            # if deterministic_error:
            #     feedback = deterministic_error
            #     continue

            code_execution = None
            if candidate.proposed_code.strip():
                code_execution = self._code_runner.run(
                    code=candidate.proposed_code,
                    tests=case.tests,
                )

            check = self._student_turn_verifier.verify(
                case=case,
                plan=plan,
                progress=progress,
                profile=profile,
                conversation=conversation,
                candidate=candidate,
                code_execution=code_execution,
            )

            if check.accepted:
                return candidate, code_execution, check

            feedback = (
                check.regeneration_feedback
                or "\n".join(check.reasons)
                or "The previous Student turn failed verification."
            )
        raise DialogueGenerationError("Could not generate an accepted Student turn.")

    def _generate_verified_tutor_turn(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        conversation: Conversation,
        tutor: OfflineTutorAgent,
        expected_learner_state: LearnerState,
        latest_code_execution: TestRunResult | None,
    ) -> tuple[TutorTurn, TutorHardCheck]:
        feedback = ""

        for _ in range(self._max_tutor_attempts):
            candidate = tutor.generate_turn(
                conversation=conversation,
                progress=progress,
                learner_state=expected_learner_state,
                latest_code_execution=latest_code_execution,
                regeneration_feedback=feedback,
            )

            deterministic_error = self._tutor_candidate_error(
                candidate=candidate,
                expected_learner_state=expected_learner_state,
                plan=plan,
                progress=progress,
                latest_code_execution=latest_code_execution,
            )

            if deterministic_error:
                feedback = deterministic_error
                continue

            hard_check = self._tutor_turn_verifier.verify(
                case=case,
                plan=plan,
                progress=progress,
                conversation=conversation,
                candidate=candidate,
                expected_learner_state=expected_learner_state,
                latest_code_execution=latest_code_execution,
            )

            if hard_check.accepted:
                return candidate, hard_check

            feedback = (
                hard_check.regeneration_feedback
                or "\n".join(hard_check.reasons)
                or "The previous Tutor turn failed hard verification."
            )

        raise DialogueGenerationError("Could not generate an accepted Tutor turn.")

    # @staticmethod
    # def _student_candidate_error(
    #     *,
    #     candidate: StudentTurn,
    #     conversation: Conversation,
    # ) -> str:
    #     if conversation.is_empty:
    #         if candidate.learner_state != LearnerState.START:
    #             return "The first Student turn must use learner_state START."

    #         return ""

    #     if candidate.learner_state == LearnerState.START:
    #         return (
    #             "START is allowed only for the first Student turn. "
    #             "Choose the state that matches the response to the Tutor."
    #         )

    #     if candidate.learner_state == LearnerState.END:
    #         return (
    #             "Do not use END for a Student response. Use CORRECT or "
    #             "COMPREHENSION when the Student has successfully understood "
    #             "the current objective."
    #         )

    #     return ""

    @staticmethod
    def _tutor_candidate_error(
        *,
        candidate: TutorTurn,
        expected_learner_state: LearnerState,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        latest_code_execution: TestRunResult | None,
    ) -> str:
        if candidate.learner_state != expected_learner_state:
            return (
                "Use the supplied verified learner state exactly. "
                f"learner_state must be {expected_learner_state.value}."
            )

        if candidate.active_step_id != progress.active_step_id:
            return (
                "Use the supplied active step. The active step is "
                f"{progress.active_step_id}."
            )

        if candidate.step_completed and candidate.learner_state not in {
            LearnerState.CORRECT,
            LearnerState.COMPREHENSION,
        }:
            return (
                "Do not mark the step complete unless the Student's response "
                "demonstrates the current objective."
            )

        final_step_id = plan.steps[-1].step_id

        if (
            candidate.step_completed
            and progress.active_step_id == final_step_id
            and (latest_code_execution is None or not latest_code_execution.passed)
        ):
            return (
                "Do not complete the final plan step yet. The Student must "
                "first submit code that passes the tests."
            )

        return ""

    @staticmethod
    def _update_progress(
        *,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        tutor_turn: TutorTurn,
    ) -> PlanProgress:
        if not tutor_turn.step_completed:
            return progress

        completed = [
            *progress.completed_step_ids,
            progress.active_step_id,
        ]

        current_index = None
        for index, step in enumerate(plan.steps):
            if step.step_id == progress.active_step_id:
                current_index = index
                break

        if current_index is None:
            raise ValueError(f"Unknown active step: {progress.active_step_id}")

        if current_index == len(plan.steps) - 1:
            return PlanProgress(
                completed_step_ids=completed,
                active_step_id=progress.active_step_id,
            )

        # Move to the next step_id
        return PlanProgress(
            completed_step_ids=completed,
            active_step_id=plan.steps[current_index + 1].step_id,
        )

    @staticmethod
    def _is_complete(
        *,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        tutor_turn: TutorTurn,
        latest_code_execution: TestRunResult | None,
    ) -> bool:
        return (
            # Conditions that are satisfied iff the tutoring dialogue has finished successfully
            len(progress.completed_step_ids) == len(plan.steps)
            and latest_code_execution is not None
            and latest_code_execution.passed
            and tutor_turn.learner_state
            in {
                LearnerState.CORRECT,
                LearnerState.COMPREHENSION,
            }
        )

    def _finalise(
        self,
        *,
        dialogue_id: str,
        case: BenchmarkCase,
        verified_plan: VerifiedPlan,
        profile: StudentProfile,
        variant: StudentVariant,
        conversation: Conversation,
        records: list[StudentTurnRecord | TutorTurnRecord],
        progress: PlanProgress,
        latest_code_execution: TestRunResult,
    ) -> GeneratedDialogue:
        completion_evidence = (
            "Completed plan steps:\n"
            f"{progress.completed_step_ids}\n\n"
            "Latest Student code execution:\n"
            f"passed={latest_code_execution.passed}\n"
            f"{latest_code_execution.output}"
        )

        verification = self._dialogue_verifier.verify(
            case=case,
            planner_output=verified_plan.output,
            student_profile=profile,
            dialogue_transcript=conversation.to_text(),
            completion_evidence=completion_evidence,
        )

        if not verification.accepted:
            raise DialogueGenerationError(
                "Completed dialogue was rejected by the final verifier: "
                f"{verification.main_issue}"
            )

        return GeneratedDialogue(
            dialogue_id=dialogue_id,
            case_id=case.case_id,
            source=case.source,
            student_variant=variant,
            student_profile=profile,
            planner_output=verified_plan.output,
            plan_verification=verified_plan.verification,
            plan_attempts=verified_plan.attempts,
            turns=records,
            dialogue_verification=verification,
        )
