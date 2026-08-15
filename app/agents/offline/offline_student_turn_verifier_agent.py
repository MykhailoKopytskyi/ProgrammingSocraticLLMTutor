from __future__ import annotations

from typing import Any

from pydantic import Field

from app.common.config import STUDENT_TURN_VERIFIER_INSTRUCTIONS

from ...common.code_runner import TestRunResult
from ...common.models import (
    BenchmarkCase,
    PedagogicalPlan,
    PlanProgress,
    StrictModel,
)
from ..agent import Agent
from .offline_student_agent import StudentProfile, StudentTurn


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
        )

    def verify(
        self,
        *,
        case: BenchmarkCase,
        plan: PedagogicalPlan,
        progress: PlanProgress,
        profile: StudentProfile,
        verified_history: str,
        candidate: StudentTurn,
        code_execution: TestRunResult | None = None,
    ) -> StudentTurnCheck:
        execution_evidence = (
            "No proposed code was executed."
            if code_execution is None
            else (f"passed={code_execution.passed}\n{code_execution.output}")
        )

        prompt = (
            "Verify the proposed Student turn.\n\n"
            "RUNTIME-VISIBLE CASE:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE:\n"
            f"{case.oracle_context()}\n\n"
            "PRIVATE STUDENT PROFILE:\n"
            f"{profile.model_dump_json(indent=2)}\n\n"
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
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=StudentTurnCheck,
        )


class StudentTurnCheck(StrictModel):
    implausible_progression: bool
    oracle_leakage: bool
    malformed_or_incoherent: bool

    reasons: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return not any(
            (
                self.implausible_progression,
                self.oracle_leakage,
                self.malformed_or_incoherent,
            )
        )
