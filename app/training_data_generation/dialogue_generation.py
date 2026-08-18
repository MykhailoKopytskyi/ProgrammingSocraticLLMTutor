from __future__ import annotations

from hashlib import sha256
from typing import Any

from ..agents.offline.offline_dialogue_verifier_agent import (
    OfflineDialogueVerifierAgent,
)
from ..agents.offline.offline_student_agent import (
    STUDENT_TURN_STATE_WEIGHTS,
    OfflineStudentAgent,
    StudentProfile,
    StudentTurn,
    StudentTurnState,
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
from ..common.models import BenchmarkCase, LearnerState, PedagogicalPlan, PlanProgress
from .models import (
    CodeExecutionRecord,
    DialogueRecords,
    GeneratedDialogue,
    StudentTurnRecord,
    TutorTurnRecord,
    VerifiedPlan,
)


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
        max_rounds: int = 14,
        max_student_attempts: int = 3,
        max_tutor_attempts: int = 3,
        student_state_seed: int = 0,
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
        self._student_state_seed = student_state_seed

    def generate(
        self,
        *,
        case: BenchmarkCase,
        verified_plan: VerifiedPlan,
        profile: StudentProfile,
        variant: StudentVariant,
    ) -> GeneratedDialogue:
        dialogue_id = f"{case.case_id}__{variant.value}"
        plan: PedagogicalPlan = verified_plan.output.plan

        student: OfflineStudentAgent = OfflineStudentAgent(
            llm=self._llm,
            model=self._student_model,
            case=case,
            profile=profile,
            variant=variant,
        )

        tutor: OfflineTutorAgent = OfflineTutorAgent(
            llm=self._llm,
            model=self._tutor_model,
            case=case,
            planner_output=verified_plan.output,
        )

        progress: PlanProgress = PlanProgress()

        dialogue_records = DialogueRecords()
        latest_code_execution: TestRunResult | None = None

        for round_index in range(self._max_rounds):
            student_state = self._student_state_for_turn(
                dialogue_id=dialogue_id,
                round_index=round_index,
            )

            (
                student_turn,
                student_execution,
                student_check,
                accepted_student_state,
            ) = self._generate_verified_student_turn(
                case=case,
                plan=plan,
                progress=progress,
                profile=profile,
                variant=variant,
                student_state=student_state,
                dialogue_id=dialogue_id,
                round_index=round_index,
                dialogue_records=dialogue_records,
                student=student,
            )
            student_record = StudentTurnRecord(
                turn=student_turn,
                student_state=accepted_student_state,
                code_execution=(
                    CodeExecutionRecord.from_result(student_execution)
                    if student_execution is not None
                    else None
                ),
                hard_check=student_check,
            )
            dialogue_records.add_student(student_record)

            if student_execution is not None:
                latest_code_execution = student_execution

            tutor_turn, hard_check = self._generate_verified_tutor_turn(
                case=case,
                plan=plan,
                progress=progress,
                dialogue_records=dialogue_records,
                tutor=tutor,
                latest_code_execution=latest_code_execution,
                learner_state=LearnerState(accepted_student_state.value),
            )
            dialogue_records.add_tutor(
                TutorTurnRecord(
                    turn=tutor_turn,
                    hard_check=hard_check,
                )
            )
            progress = self._update_progress(
                tutor_turn=tutor_turn,
            )

            if self._is_complete(
                plan=plan,
                progress=progress,
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
                    dialogue_records=dialogue_records,
                    progress=progress,
                    latest_code_execution=latest_code_execution,
                )

        raise DialogueGenerationError(
            "Conversation reached the maximum number of rounds without valid completion."
        )

    def _student_state_for_turn(
        self,
        *,
        dialogue_id: str,
        round_index: int,
        resample_index: int = 0,
        excluded_states: tuple[StudentTurnState, ...] = (),
    ) -> StudentTurnState:
        if round_index == 0:
            return StudentTurnState.START

        if round_index >= 8:
            states = (
                StudentTurnState.CORRECT,
                StudentTurnState.COMPREHENSION,
            )
        else:
            states = (
                StudentTurnState.CORRECT,
                StudentTurnState.INCORRECT,
                StudentTurnState.QUESTION,
                StudentTurnState.COMPREHENSION,
                StudentTurnState.CONFUSION,
                StudentTurnState.IRRELEVANT,
            )

        available_states = tuple(
            state for state in states if state not in excluded_states
        )
        if available_states:
            states = available_states

        if resample_index == 0:
            key_text = f"{self._student_state_seed}:{dialogue_id}:{round_index}"
        else:
            key_text = (
                f"{self._student_state_seed}:{dialogue_id}:{round_index}:"
                f"resample:{resample_index}"
            )
        key = key_text.encode("utf-8")
        digest = sha256(key).digest()
        sample = int.from_bytes(digest[:8], "big") / float(2**64)

        total_weight = 0.0
        for state in states:
            total_weight += STUDENT_TURN_STATE_WEIGHTS[state]

        cumulative = 0.0
        for state in states:
            cumulative += STUDENT_TURN_STATE_WEIGHTS[state] / total_weight
            if sample < cumulative:
                return state

        return states[-1]

    def _generate_verified_student_turn(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        profile: StudentProfile,
        variant: StudentVariant,
        student_state: StudentTurnState,
        dialogue_id: str,
        round_index: int,
        dialogue_records: DialogueRecords,
        student: OfflineStudentAgent,
    ) -> tuple[
        StudentTurn,
        TestRunResult | None,
        StudentTurnCheck,
        StudentTurnState,
    ]:
        feedback = ""
        current_state = student_state
        state_only_failures = 0

        for attempt in range(1, self._max_student_attempts + 1):
            candidate = student.generate_turn(
                dialogue_history=dialogue_records.to_student_text(),
                is_first_turn=dialogue_records.is_empty,
                turn_state=current_state,
                regeneration_feedback=feedback,
            )

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
                variant=variant,
                turn_state=current_state,
                verified_history=dialogue_records.to_tutor_text(),
                candidate=candidate,
                code_execution=code_execution,
            )

            if check.accepted:
                return candidate, code_execution, check, current_state

            state_only_failure = (
                not check.state_consistent
                and not check.implausible_progression
                and not check.oracle_leakage
                and not check.malformed_or_incoherent
            )

            if state_only_failure:
                state_only_failures += 1
            else:
                state_only_failures = 0

            # If the same sampled state cannot be realized twice, use the final
            # attempt with a newly sampled state. This prevents state-specific
            # regeneration from repeatedly fighting an awkward target while
            # keeping all retry loops bounded.
            if (
                state_only_failures >= 2
                and attempt < self._max_student_attempts
                and current_state != StudentTurnState.START
            ):
                previous_state = current_state
                current_state = self._student_state_for_turn(
                    dialogue_id=dialogue_id,
                    round_index=round_index,
                    resample_index=1,
                    excluded_states=(previous_state,),
                )
                print(
                    "  Student state resampled after repeated state mismatch: "
                    f"{previous_state.value} -> {current_state.value}"
                )
                feedback = ""
                state_only_failures = 0
                continue

            feedback = self._student_regeneration_feedback(
                check,
                variant=variant,
                student_state=current_state,
            )

        raise DialogueGenerationError(
            "Could not generate an accepted Student turn within the bounded "
            f"{self._max_student_attempts} attempts."
        )

    @staticmethod
    def _student_regeneration_feedback(
        check: StudentTurnCheck,
        *,
        variant: StudentVariant,
        student_state: StudentTurnState,
    ) -> str:
        """Return oracle-safe feedback that cannot leak hidden answers."""
        issues: list[str] = []

        state_feedback = {
            StudentTurnState.START: (
                "The target learner state is START: give only an opening "
                "help-seeking Student turn and do not solve the bug yet."
            ),
            StudentTurnState.CORRECT: (
                "The target learner state is CORRECT: directly and correctly "
                "answer only the Tutor's current request."
            ),
            StudentTurnState.INCORRECT: (
                "The target learner state is INCORRECT: give a plausible, "
                "materially wrong attempt at the current unresolved request. "
                "Do not give the correct answer or a correct passing repair."
            ),
            StudentTurnState.QUESTION: (
                "The target learner state is QUESTION: ask a relevant technical "
                "clarification instead of answering the current request."
            ),
            StudentTurnState.COMPREHENSION: (
                "The target learner state is COMPREHENSION: demonstrate genuine "
                "understanding of the current concept in your own words. Stay focused "
                "on the current request; incidental evidence relevant to later work "
                "is allowed."
            ),
            StudentTurnState.CONFUSION: (
                "The target learner state is CONFUSION: express a relevant lack "
                "of understanding and ask for simpler guidance."
            ),
            StudentTurnState.IRRELEVANT: (
                "The target learner state is IRRELEVANT: give a brief off-topic "
                "reply and do not include solution progress."
            ),
        }

        if not check.state_consistent:
            issues.append(state_feedback[student_state])

        if check.implausible_progression:
            issues.append(
                "Stay focused on the Tutor's current objective. Do not deliberately "
                "abandon it for unrelated later work, submit code before implementation, "
                "revision, application, or verification is requested, or make unrelated "
                "semantic changes. Incidental later-objective evidence is allowed."
            )

        if check.oracle_leakage:
            issues.append(
                "Respond only from information available to the Student. Do not "
                "mention or rely on private, oracle, verifier, or hidden-plan information."
            )

        if check.malformed_or_incoherent:
            issues.append(
                "Produce one coherent Student response. Include proposed_code only "
                "when attempting a complete program revision."
            )

        return (
            "\n".join(issues)
            or "Generate a Student response that satisfies the checks."
        )

    def _generate_verified_tutor_turn(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        dialogue_records: DialogueRecords,
        tutor: OfflineTutorAgent,
        learner_state: LearnerState,
        latest_code_execution: TestRunResult | None,
    ) -> tuple[TutorTurn, TutorHardCheck]:
        feedback = ""
        if dialogue_records.pending_student is None:
            raise ValueError("Tutor requires a pending verified Student turn.")

        for attempt in range(1, self._max_tutor_attempts + 1):
            candidate = tutor.generate_turn(
                verified_history=dialogue_records.to_tutor_text(),
                progress=progress,
                learner_state=learner_state,
                latest_code_execution=latest_code_execution,
                regeneration_feedback=feedback,
            )
            candidate.learner_state = learner_state

            deterministic_error = self._tutor_candidate_error(
                candidate=candidate,
                plan=plan,
                progress=progress,
                latest_code_execution=latest_code_execution,
            )

            if deterministic_error:
                print(f"  Tutor attempt {attempt} deterministic rejection:")
                print(f"  {deterministic_error}")
                feedback = deterministic_error
                continue

            hard_check = self._tutor_turn_verifier.verify(
                case=case,
                plan=plan,
                progress=progress,
                candidate=candidate,
                verified_history=dialogue_records.to_tutor_text(),
                latest_code_execution=latest_code_execution,
            )

            if hard_check.accepted:
                return candidate, hard_check

            print(f"  Tutor attempt {attempt} verifier rejection:")
            print(f"  {hard_check.model_dump_json(indent=2)}")

            feedback = (
                hard_check.regeneration_feedback
                or "\n".join(hard_check.reasons)
                or "The previous Tutor turn failed hard verification."
            )

        raise DialogueGenerationError(
            f"Could not generate an accepted Tutor turn. Last feedback: {feedback}"
        )

    @staticmethod
    def _tutor_candidate_error(
        *,
        candidate: TutorTurn,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        latest_code_execution: TestRunResult | None,
    ) -> str | None:
        step_ids = [step.step_id for step in plan.steps]

        previous_completed_index = (
            -1
            if progress.completed_through_step_id is None
            else step_ids.index(progress.completed_through_step_id)
        )

        if candidate.completed_through_step_id is None:
            if previous_completed_index >= 0:
                return (
                    "completed_through_step_id is cumulative and cannot become "
                    "null after earlier plan objectives were completed."
                )
            return None

        if candidate.completed_through_step_id not in step_ids:
            return "completed_through_step_id must be a valid plan step."

        completed_through_index = step_ids.index(candidate.completed_through_step_id)

        if completed_through_index < previous_completed_index:
            return "completed_through_step_id is cumulative and cannot move backwards."

        made_new_progress = completed_through_index > previous_completed_index

        if made_new_progress and candidate.learner_state not in {
            LearnerState.CORRECT,
            LearnerState.COMPREHENSION,
        }:
            return (
                "Do not advance completed_through_step_id unless the latest "
                "Student turn demonstrates sufficient correctness or "
                "understanding."
            )

        if completed_through_index == len(plan.steps) - 1 and (
            latest_code_execution is None or not latest_code_execution.passed
        ):
            return (
                "Do not complete through the final plan step until the Student "
                "has submitted code that passes the tests."
            )

        return None

    @staticmethod
    def _update_progress(
        *,
        tutor_turn: TutorTurn,
    ) -> PlanProgress:
        return PlanProgress(
            completed_through_step_id=tutor_turn.completed_through_step_id
        )

    @staticmethod
    def _is_complete(
        *,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        latest_code_execution: TestRunResult | None,
    ) -> bool:
        return (
            # Conditions that are satisfied iff the tutoring dialogue has finished successfully
            progress.completed_through_step_id == plan.steps[-1].step_id
            and latest_code_execution is not None
            and latest_code_execution.passed
        )

    def _finalise(
        self,
        *,
        dialogue_id: str,
        case: BenchmarkCase,
        verified_plan: VerifiedPlan,
        profile: StudentProfile,
        variant: StudentVariant,
        dialogue_records: DialogueRecords,
        progress: PlanProgress,
        latest_code_execution: TestRunResult,
    ) -> GeneratedDialogue:
        completion_evidence = (
            "Completed through plan step:\n"
            f"{progress.completed_through_step_id}\n\n"
            "Latest Student code execution:\n"
            f"passed={latest_code_execution.passed}\n"
            f"{latest_code_execution.output}"
        )

        verification = self._dialogue_verifier.verify(
            case=case,
            planner_output=verified_plan.output,
            student_profile=profile,
            student_variant=variant,
            dialogue_transcript=dialogue_records.to_verifier_text(),
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
            verified_plan=verified_plan,
            records=dialogue_records,
            dialogue_verification=verification,
        )
