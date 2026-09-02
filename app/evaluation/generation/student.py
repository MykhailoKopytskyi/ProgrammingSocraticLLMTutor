from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock

from app.agents.offline.offline_student_agent import (
    STUDENT_TURN_STATE_INSTRUCTIONS,
    STUDENT_TURN_STATE_WEIGHTS_BY_VARIANT,
    STUDENT_VARIANT_INSTRUCTIONS,
    StudentProfile,
    StudentTurn,
    StudentVariant,
)
from app.agents.offline.offline_student_turn_verifier_agent import StudentTurnAssessment
from app.common.code_runner import CodeRunner, TestRunResult
from app.common.config import (
    OFFLINE_STUDENT_AGENT_INSTRUCTIONS,
    OFFLINE_STUDENT_PROFILE_INSTRUCTIONS,
    STUDENT_TURN_VERIFIER_INSTRUCTIONS,
)
from app.common.models import BenchmarkCase, LearnerState

from ..backends import StructuredModelBackend
from ..conversation import ConversationRenderer
from ..models import EvaluationRound, EvaluationStudentTurn
from ..storage import JsonlStore


class StudentGenerationError(RuntimeError):
    """A controlled Student turn could not be produced under the required protocol."""


class StudentProfileStore:
    def __init__(
        self,
        *,
        path: str | Path,
        backend: StructuredModelBackend,
    ):
        self.path = Path(path)
        self.backend = backend
        self.store = JsonlStore(self.path)
        self._profiles = self._load()
        self._state_lock = Lock()
        self._write_lock = Lock()
        self._case_locks: dict[str, Lock] = {}

    @property
    def config_id(self) -> str:
        return f"{self.backend.config_id}:student-profile"

    def get(self, case: BenchmarkCase) -> StudentProfile:
        case_lock = self._case_lock(case.case_id)
        with case_lock:
            with self._state_lock:
                existing = self._profiles.get(case.case_id)
            if existing is not None:
                return existing

            prompt = (
                "Generate the private student belief profile.\n\n"
                "RUNTIME-VISIBLE CASE:\n"
                f"{case.visible_context()}\n\n"
                "TRAINING-ONLY ORACLE:\n"
                f"{case.oracle_context()}"
            )
            result = self.backend.generate(
                system_prompt=OFFLINE_STUDENT_PROFILE_INSTRUCTIONS,
                user_prompt=prompt,
                output_type=StudentProfile,
                max_output_tokens=1500,
            )
            profile = result.parsed
            with self._state_lock:
                self._profiles[case.case_id] = profile
            with self._write_lock:
                self._append(case.case_id, profile)
            return profile

    def _case_lock(self, case_id: str) -> Lock:
        with self._state_lock:
            lock = self._case_locks.get(case_id)
            if lock is None:
                lock = Lock()
                self._case_locks[case_id] = lock
            return lock

    def _load(self) -> dict[str, StudentProfile]:
        result: dict[str, StudentProfile] = {}
        for row in self.store.read_mappings():
            if row.get("profile_config_id") != self.config_id:
                continue
            try:
                case_id = str(row["case_id"])
                profile = StudentProfile.model_validate(row["profile"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid Student profile record in {self.path}") from error
            result[case_id] = profile
        return result

    def _append(self, case_id: str, profile: StudentProfile) -> None:
        self.store.append_mapping(
            {
                "case_id": case_id,
                "profile_config_id": self.config_id,
                "profile": profile.model_dump(mode="json"),
            }
        )


class StudentSystem(ABC):
    @property
    @abstractmethod
    def system_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def start_case(self, case: BenchmarkCase, variant: StudentVariant) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_turn(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        target_state: LearnerState,
        is_first_turn: bool,
    ) -> tuple[EvaluationStudentTurn, TestRunResult | None]:
        raise NotImplementedError


class ControlledStudentSystem(StudentSystem):
    def __init__(
        self,
        *,
        generator: StructuredModelBackend,
        verifier: StructuredModelBackend,
        profile_store: StudentProfileStore,
        code_runner: CodeRunner,
        max_attempts: int = 3,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.generator = generator
        self.verifier = verifier
        self.profile_store = profile_store
        self.code_runner = code_runner
        self.max_attempts = max_attempts
        self.case: BenchmarkCase | None = None
        self.variant: StudentVariant | None = None
        self.profile: StudentProfile | None = None

    @property
    def system_id(self) -> str:
        return f"controlled_student__{self.generator.backend_id}"

    @property
    def config_id(self) -> str:
        return (
            f"generator={self.generator.config_id}|"
            f"verifier={self.verifier.config_id}|"
            f"profile={self.profile_store.config_id}|"
            f"max_attempts={self.max_attempts}"
        )

    def start_case(self, case: BenchmarkCase, variant: StudentVariant) -> None:
        self.case = case
        self.variant = variant
        self.profile = self.profile_store.get(case)

    def generate_turn(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        target_state: LearnerState,
        is_first_turn: bool,
    ) -> tuple[EvaluationStudentTurn, TestRunResult | None]:
        if self.case is None or self.variant is None or self.profile is None:
            raise RuntimeError("Student system has not been started")

        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            candidate = self._generate_candidate(
                history=history,
                target_state=target_state,
                is_first_turn=is_first_turn,
                feedback=feedback,
            )
            execution = None
            if candidate.proposed_code.strip():
                execution = self.code_runner.run(
                    code=candidate.proposed_code,
                    tests=self.case.tests,
                )

            assessment = self._verify_candidate(
                history=history,
                target_state=target_state,
                candidate=candidate,
                execution=execution,
                previous_feedback=feedback,
            )
            accepted = assessment.state_consistent and not any(
                (
                    assessment.implausible_progression,
                    assessment.oracle_leakage,
                    assessment.malformed_or_incoherent,
                )
            )
            if accepted:
                return (
                    EvaluationStudentTurn(
                        target_learner_state=target_state,
                        reply=candidate.reply,
                        proposed_code=candidate.proposed_code,
                        generation_attempts=attempt,
                    ),
                    execution,
                )

            feedback = self._regeneration_feedback(
                assessment=assessment,
                target_state=target_state,
            )

        raise StudentGenerationError(
            f"Could not generate verified {target_state.value} Student turn "
            f"within {self.max_attempts} attempts"
        )

    def _generate_candidate(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        target_state: LearnerState,
        is_first_turn: bool,
        feedback: str,
    ) -> StudentTurn:
        assert self.case is not None
        assert self.variant is not None
        assert self.profile is not None

        system_prompt = (
            "# Private reference information\n\n"
            "The following information is private grounding for controlled Student "
            "simulation. It contains the benchmark oracle. Use it internally to realise "
            "the sampled learner state accurately. Never mention that this private "
            "information was supplied, and do not expose future answers unless the current "
            "Tutor request and sampled learner state require them.\n\n"
            "For CORRECT and COMPREHENSION, use this reference information to make the "
            "response technically correct for the Tutor's current request. For INCORRECT, "
            "use it to construct a plausible materially wrong response rather than "
            "accidentally giving the correct answer.\n\n"
            "<reference_information>\n"
            f"{self.case.oracle_context()}\n"
            "</reference_information>\n\n"
            f"{OFFLINE_STUDENT_AGENT_INSTRUCTIONS}\n\n"
            "# Private student profile for this conversation\n\n"
            "The following profile defines the Student's stable learner background "
            "and the beliefs or assumptions they currently hold. Use the background "
            "fields to keep vocabulary, confidence, and assumed programming knowledge "
            "appropriate for this learner. Treat the beliefs as genuine beliefs; do "
            "not infer that they are correct or incorrect merely because they appear "
            "in the profile. Use the beliefs to ground the Student's reasoning when "
            "compatible with the sampled learner state.\n\n"
            "<student_profile>\n"
            f"{self.profile.model_dump_json(indent=2)}\n"
            "</student_profile>\n\n"
            "# Private behaviour tendency\n\n"
            f"Variant: {self.variant.value.upper()}\n\n"
            f"{STUDENT_VARIANT_INSTRUCTIONS[self.variant]}\n\n"
            "The sampled learner state is authoritative for every turn. "
            "StudentProfile and behaviour tendency influence the Student's reasoning, "
            "background, confidence, and style, but never override the semantic "
            "requirements of the sampled learner state."
        )

        task = (
            "Generate the first Student turn. Briefly describe the observed "
            "difficulty with the buggy program and ask the Tutor for help. "
            "Do not provide corrected code or an exact final repair on this first turn, "
            "and do not fully solve the debugging problem before tutoring has begun."
            if is_first_turn
            else (
                "Generate one Student turn that realizes the supplied target learner "
                "state. Use the Tutor's latest message only to identify the current "
                "topic or task; the sampled state controls the semantic behaviour."
            )
        )

        user_prompt = (
            f"{task}\n\n"
            "<target_learner_state>\n"
            f"State: {target_state.value}\n"
            f"{STUDENT_TURN_STATE_INSTRUCTIONS[target_state]}\n"
            "</target_learner_state>\n\n"
            "<programming_case>\n"
            f"{self.case.visible_context()}\n"
            "</programming_case>\n\n"
            "<conversation_history>\n"
            f"{ConversationRenderer.render(history)}\n"
            "</conversation_history>"
        )
        if feedback.strip():
            user_prompt += (
                "\n\n<regeneration_feedback>\n"
                f"{feedback.strip()}\n"
                "</regeneration_feedback>"
            )

        result = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=StudentTurn,
            max_output_tokens=2000,
        )
        return result.parsed

    def _regeneration_feedback(
        self,
        *,
        assessment: StudentTurnAssessment,
        target_state: LearnerState,
    ) -> str:
        state_feedback = {
            LearnerState.START: (
                "The target learner state is START: give only an opening help-seeking "
                "Student turn and do not solve the bug yet."
            ),
            LearnerState.CORRECT: (
                "The target learner state is CORRECT: directly and correctly answer "
                "only the Tutor's current request."
            ),
            LearnerState.INCORRECT: (
                "The target learner state is INCORRECT: give a plausible, materially "
                "wrong attempt at the current unresolved request. Do not give the "
                "correct answer or a correct passing repair."
            ),
            LearnerState.QUESTION: (
                "The target learner state is QUESTION: ask a relevant technical "
                "clarification instead of answering the current request."
            ),
            LearnerState.COMPREHENSION: (
                "The target learner state is COMPREHENSION: demonstrate genuine "
                "understanding of the current concept in your own words. Stay focused "
                "on the current request; incidental evidence relevant to later work "
                "is allowed."
            ),
            LearnerState.CONFUSION: (
                "The target learner state is CONFUSION: express a relevant lack of "
                "understanding and ask for simpler guidance."
            ),
            LearnerState.IRRELEVANT: (
                "The target learner state is IRRELEVANT: give a brief off-topic reply "
                "and do not include solution progress."
            ),
        }

        issues = []
        if assessment.reasons:
            issues.append(
                "Verifier feedback from the previous rejected Student candidate:\n"
                + "\n".join(assessment.reasons)
            )
        if not assessment.state_consistent:
            issues.append(state_feedback[target_state])
        if assessment.implausible_progression:
            issues.append(
                "Stay focused on the Tutor's current objective. Do not deliberately "
                "abandon it for unrelated later work, submit code before implementation, "
                "revision, application, or verification is requested, or make unrelated "
                "semantic changes. Incidental later-objective evidence is allowed."
            )
        if assessment.oracle_leakage:
            issues.append(
                "Do not expose or refer to the private oracle, expected answers, "
                "verifier feedback, or other generation metadata in the visible Student "
                "response. Use private reference information only as internal grounding."
            )
        if assessment.malformed_or_incoherent:
            issues.append(
                "Produce one coherent Student response. Include proposed_code only when "
                "attempting a complete program revision."
            )
        return "\n\n".join(issues)

    def _verify_candidate(
        self,
        *,
        history: tuple[EvaluationRound, ...],
        target_state: LearnerState,
        candidate: StudentTurn,
        execution: TestRunResult | None,
        previous_feedback: str,
    ) -> StudentTurnAssessment:
        assert self.case is not None
        assert self.variant is not None
        assert self.profile is not None

        execution_text = (
            "No proposed code was executed."
            if execution is None
            else f"passed={execution.passed}\n{execution.output}"
        )
        user_prompt = (
            "Verify the proposed Student turn against its sampled learner state.\n\n"
            "TARGET LEARNER STATE:\n"
            f"{target_state.value}\n\n"
            "RUNTIME-VISIBLE CASE:\n"
            f"{self.case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE:\n"
            f"{self.case.oracle_context()}\n\n"
            "PRIVATE STUDENT PROFILE:\n"
            f"{self.profile.model_dump_json(indent=2)}\n\n"
            "INTENDED STUDENT VARIANT:\n"
            f"{self.variant.value.upper()}\n"
            f"{STUDENT_VARIANT_INSTRUCTIONS[self.variant]}\n\n"
            "ACCEPTED CONVERSATION BEFORE THIS STUDENT TURN:\n"
            f"{ConversationRenderer.render(history)}\n\n"
            "CANDIDATE STUDENT TURN:\n"
            f"{candidate.model_dump_json(indent=2)}\n\n"
            "CANDIDATE CODE EXECUTION:\n"
            f"{execution_text}\n\n"
            "PREVIOUS REGENERATION FEEDBACK:\n"
            f"{previous_feedback or '[none]'}"
        )
        result = self.verifier.generate(
            system_prompt=STUDENT_TURN_VERIFIER_INSTRUCTIONS,
            user_prompt=user_prompt,
            output_type=StudentTurnAssessment,
            max_output_tokens=3000,
        )
        return result.parsed


class StudentStateSampler:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def state_for_turn(
        self,
        *,
        case_id: str,
        variant: StudentVariant,
        round_index: int,
    ) -> LearnerState:
        if round_index == 0:
            return LearnerState.START

        if round_index >= 8:
            states = (LearnerState.CORRECT, LearnerState.COMPREHENSION)
        else:
            states = (
                LearnerState.CORRECT,
                LearnerState.INCORRECT,
                LearnerState.QUESTION,
                LearnerState.COMPREHENSION,
                LearnerState.CONFUSION,
                LearnerState.IRRELEVANT,
            )

        dialogue_id = f"{case_id}__{variant.value}"
        rng = random.Random(f"{self.seed}:{dialogue_id}:{round_index}")
        sample = rng.random()

        weights = STUDENT_TURN_STATE_WEIGHTS_BY_VARIANT[variant]
        total = sum(weights[state] for state in states)
        cumulative = 0.0
        for state in states:
            cumulative += weights[state] / total
            if sample < cumulative:
                return state
        return states[-1]
