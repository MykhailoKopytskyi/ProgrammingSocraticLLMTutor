from __future__ import annotations

from ..agents.offline.offline_dialogue_verifier_agent import (
    OfflineDialogueVerifierAgent,
)
from ..agents.offline.offline_turn_verifier_agent import (
    OfflineTurnVerifierAgent,
    TutorHardCheck,
)
from ..agents.student_agent import LearnerState, StudentAgent, StudentProfile
from ..agents.tutor_agent import TutorAgent, TutorTurn
from ..common.conversation import Conversation
from ..common.models import BenchmarkCase, PedagogicalPlan, PlanProgress
from ..execution.code_runner import CodeRunner, TestRunResult
from .dialogue_models import (
    CodeExecutionRecord,
    GeneratedDialogue,
    StudentTurnRecord,
    TutorTurnRecord,
)
from .plan_pipeline import VerifiedPlan


class DialogueGenerationError(RuntimeError):
    pass


class DialogueGenerationPipeline:
    def __init__(
        self,
        *,
        turn_verifier: OfflineTurnVerifierAgent,
        dialogue_verifier: OfflineDialogueVerifierAgent,
        code_runner: CodeRunner,
        max_rounds: int = 12,
        max_tutor_attempts: int = 3,
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

        if max_tutor_attempts < 1:
            raise ValueError("max_tutor_attempts must be at least 1")

        self._turn_verifier = turn_verifier
        self._dialogue_verifier = dialogue_verifier
        self._code_runner = code_runner
        self._max_rounds = max_rounds
        self._max_tutor_attempts = max_tutor_attempts

    def generate(
        self,
        *,
        case: BenchmarkCase,
        verified_plan: VerifiedPlan,
        profile: StudentProfile,
        student_agent: StudentAgent,
        tutor_agent: TutorAgent,
    ) -> GeneratedDialogue:
        plan = verified_plan.output.plan
        progress = PlanProgress(
            completed_step_ids=[],
            active_step_id=plan.steps[0].step_id,
        )
        conversation = Conversation()
        records: list[StudentTurnRecord | TutorTurnRecord] = []
        latest_code_execution: TestRunResult | None = None

        for _ in range(self._max_rounds):
            student_turn = student_agent.generate_turn(conversation)
            student_message = student_turn.to_message()
            conversation.add(student_message)

            student_execution: TestRunResult | None = None
            if student_turn.proposed_code.strip():
                student_execution = self._code_runner.run(
                    code=student_turn.proposed_code,
                    tests=case.tests,
                )
                latest_code_execution = student_execution

            records.append(
                StudentTurnRecord(
                    turn=student_turn,
                    code_execution=(
                        CodeExecutionRecord.from_result(latest_code_execution)
                        if (
                            student_turn.proposed_code.strip()
                            and latest_code_execution is not None
                        )
                        else None
                    ),
                )
            )

            tutor_turn, hard_check = self._generate_verified_tutor_turn(
                case=case,
                plan=plan,
                progress=progress,
                conversation=conversation,
                tutor_agent=tutor_agent,
                latest_code_execution=latest_code_execution,
            )

            tutor_message = tutor_turn.to_message()
            conversation.add(tutor_message)
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
                return self._finalize(
                    case=case,
                    verified_plan=verified_plan,
                    profile=profile,
                    conversation=conversation,
                    records=records,
                    progress=progress,
                    latest_code_execution=latest_code_execution,
                )

        raise DialogueGenerationError(
            "Conversation reached the maximum number of rounds without valid completion."
        )

    def _generate_verified_tutor_turn(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        conversation: Conversation,
        tutor_agent: TutorAgent,
        latest_code_execution: TestRunResult | None,
    ) -> tuple[TutorTurn, TutorHardCheck]:
        feedback = ""

        for _ in range(self._max_tutor_attempts):
            candidate = tutor_agent.generate_turn(
                conversation=conversation,
                progress=progress,
                regeneration_feedback=feedback,
            )

            deterministic_error = self._candidate_error(
                candidate=candidate,
                plan=plan,
                progress=progress,
                latest_code_execution=latest_code_execution,
            )

            if deterministic_error:
                feedback = deterministic_error
                continue

            hard_check = self._turn_verifier.verify(
                case=case,
                plan=plan,
                progress=progress,
                conversation=conversation,
                candidate=candidate,
            )

            if hard_check.accepted:
                return candidate, hard_check

            feedback = (
                hard_check.regeneration_feedback
                or "\n".join(hard_check.reasons)
                or "The previous Tutor turn failed hard verification."
            )

        raise DialogueGenerationError("Could not generate an accepted Tutor turn.")

    @staticmethod
    def _candidate_error(
        *,
        candidate: TutorTurn,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        latest_code_execution: TestRunResult | None,
    ) -> str:
        if candidate.active_step_id != progress.active_step_id:
            return (
                "Use the supplied active step. The active step is "
                f"{progress.active_step_id}."
            )

        if candidate.step_completed and candidate.learner_state not in {
            LearnerState.CORRECT,
            LearnerState.COMPREHENSION,
            LearnerState.END,
        }:
            return (
                "Do not mark the step complete unless the student's response "
                "demonstrates the objective."
            )

        final_step_id = plan.steps[-1].step_id
        if (
            candidate.step_completed
            and progress.active_step_id == final_step_id
            and (latest_code_execution is None or not latest_code_execution.passed)
        ):
            return (
                "Do not complete the final plan step yet. The student must first "
                "submit code that passes the tests."
            )

        if candidate.learner_state == LearnerState.END and (
            latest_code_execution is None or not latest_code_execution.passed
        ):
            return "END is not allowed before the student's code passes the tests."

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

        completed = [*progress.completed_step_ids, progress.active_step_id]
        current_index = next(
            index
            for index, step in enumerate(plan.steps)
            if step.step_id == progress.active_step_id
        )

        if current_index == len(plan.steps) - 1:
            return PlanProgress(
                completed_step_ids=completed,
                active_step_id=progress.active_step_id,
            )

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
            len(progress.completed_step_ids) == len(plan.steps)
            and latest_code_execution is not None
            and latest_code_execution.passed
            and tutor_turn.learner_state
            in {
                LearnerState.CORRECT,
                LearnerState.COMPREHENSION,
                LearnerState.END,
            }
        )

    def _finalize(
        self,
        *,
        case: BenchmarkCase,
        verified_plan: VerifiedPlan,
        profile: StudentProfile,
        conversation: Conversation,
        records: list[StudentTurnRecord | TutorTurnRecord],
        progress: PlanProgress,
        latest_code_execution: TestRunResult,
    ) -> GeneratedDialogue:
        completion_evidence = (
            "Completed plan steps:\n"
            f"{progress.completed_step_ids}\n\n"
            "Latest student code execution:\n"
            f"passed={latest_code_execution.passed}\n"
            f"{latest_code_execution.output}"
        )

        verification = self._dialogue_verifier.verify(
            case=case,
            planner_output=verified_plan.output,
            dialogue_transcript=conversation.to_text(),
            completion_evidence=completion_evidence,
        )

        if not verification.accepted:
            raise DialogueGenerationError(
                "Completed dialogue was rejected by the final verifier: "
                f"{verification.main_issue}"
            )

        return GeneratedDialogue(
            case_id=case.case_id,
            source=case.source,
            student_profile=profile,
            planner_output=verified_plan.output,
            turns=records,
            dialogue_verification=verification,
        )
