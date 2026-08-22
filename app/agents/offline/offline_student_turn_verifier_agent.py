from __future__ import annotations

from typing import Any

from pydantic import Field

from app.common.config import STUDENT_TURN_VERIFIER_INSTRUCTIONS

from ...common.code_runner import TestRunResult
from ...common.models import (
    BenchmarkCase,
    LearnerState,
    PedagogicalPlan,
    PlanProgress,
    StrictModel,
)
from ..agent import Agent
from .offline_student_agent import (
    STUDENT_VARIANT_INSTRUCTIONS,
    StudentProfile,
    StudentTurn,
    StudentTurnState,
    StudentVariant,
)


class StudentTurnAssessment(StrictModel):
    """LLM assessment used to build the deterministic stored hard check."""

    state_consistent: bool
    implausible_progression: bool
    oracle_leakage: bool
    malformed_or_incoherent: bool
    reasons: list[str] = Field(default_factory=list)


class StudentTurnCheck(StrictModel):
    """Final deterministic check used by generation and stored in data."""

    # Explicit final decision. Defaults keep older accepted records loadable.
    accepted: bool = True
    state_consistent: bool = True

    implausible_progression: bool
    oracle_leakage: bool
    malformed_or_incoherent: bool
    reasons: list[str] = Field(default_factory=list)

    # Legacy audit fields retained only so previously generated JSON can load.
    target_learner_state: LearnerState | None = None
    assessed_learner_state: LearnerState | None = None
    state_matches_target: bool | None = None


class OfflineStudentTurnVerifierAgent(Agent):
    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = STUDENT_TURN_VERIFIER_INSTRUCTIONS,
    ):
        super().__init__(
            llm=llm,
            model=model,
            instructions=instructions,
            reasoning_effort="medium",
            max_output_tokens=4000,
        )

    def verify(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        profile: StudentProfile,
        variant: StudentVariant,
        turn_state: StudentTurnState,
        verified_history: str,
        candidate: StudentTurn,
        code_execution: TestRunResult | None = None,
        previous_regeneration_feedback="",
    ) -> StudentTurnCheck:
        execution_evidence = (
            "No proposed code was executed."
            if code_execution is None
            else (f"passed={code_execution.passed}\n{code_execution.output}")
        )

        prompt = (
            "Verify the proposed Student turn against its sampled learner state.\n\n"
            "TARGET LEARNER STATE:\n"
            f"{turn_state.value}\n\n"
            "RUNTIME-VISIBLE CASE:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE:\n"
            f"{case.oracle_context()}\n\n"
            "PRIVATE STUDENT PROFILE:\n"
            f"{profile.model_dump_json(indent=2)}\n\n"
            "INTENDED STUDENT VARIANT:\n"
            f"{variant.value.upper()}\n"
            f"{STUDENT_VARIANT_INSTRUCTIONS[variant]}\n\n"
            "FIXED PEDAGOGICAL PLAN:\n"
            f"{plan.model_dump_json(indent=2)}\n\n"
            "CURRENT PLAN PROGRESS:\n"
            f"{progress.model_dump_json(indent=2)}\n\n"
            "ACCEPTED CONVERSATION BEFORE THIS STUDENT TURN:\n"
            f"{verified_history}\n\n"
            "CANDIDATE STUDENT TURN:\n"
            f"{candidate.model_dump_json(indent=2)}\n\n"
            "CANDIDATE CODE EXECUTION:\n"
            f"{execution_evidence}"
            "PREVIOUS REGENERATION FEEDBACK GIVEN TO THIS STUDENT:\n"
            f"{previous_regeneration_feedback or '[none]'}\n\n"
        )

        assessment = self._get_structured_output(
            prompt=prompt,
            output_type=StudentTurnAssessment,
        )

        accepted = assessment.state_consistent and not any(
            (
                assessment.implausible_progression,
                assessment.oracle_leakage,
                assessment.malformed_or_incoherent,
            )
        )

        return StudentTurnCheck(
            accepted=accepted,
            state_consistent=assessment.state_consistent,
            target_learner_state=turn_state,
            assessed_learner_state=None,
            state_matches_target=assessment.state_consistent,
            implausible_progression=assessment.implausible_progression,
            oracle_leakage=assessment.oracle_leakage,
            malformed_or_incoherent=assessment.malformed_or_incoherent,
            reasons=assessment.reasons,
        )
