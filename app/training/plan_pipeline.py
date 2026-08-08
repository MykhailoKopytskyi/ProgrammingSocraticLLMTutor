from __future__ import annotations

from dataclasses import dataclass

from ..agents.offline.offline_plan_verifier_agent import (
    OfflinePlanVerifierAgent,
    PlanVerification,
)
from ..agents.offline.offline_planner_agent import OfflinePlannerAgent
from ..common.models import BenchmarkCase, PlannerOutput
from ..execution.code_runner import CodeRunner


class PlanGenerationError(RuntimeError):
    """Raised when no acceptable plan is produced within the retry limit."""


@dataclass(frozen=True)
class VerifiedPlan:
    output: PlannerOutput
    verification: PlanVerification
    attempts: int


class PlanningPipeline:
    def __init__(
        self,
        *,
        planner: OfflinePlannerAgent,
        verifier: OfflinePlanVerifierAgent,
        code_runner: CodeRunner,
        max_attempts: int = 3,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._planner = planner
        self._verifier = verifier
        self._code_runner = code_runner
        self._max_attempts = max_attempts

    def generate(
        self,
        case: BenchmarkCase,
    ) -> VerifiedPlan:
        regeneration_feedback = ""
        last_error = ""

        for attempt in range(1, self._max_attempts + 1):
            planner_output = self._planner.generate_output(
                case=case,
                regeneration_feedback=regeneration_feedback,
            )

            execution = self._code_runner.run(
                code=planner_output.corrected_code,
                tests=case.tests,
            )

            if not execution.passed:
                last_error = execution.output
                regeneration_feedback = (
                    "Your candidate corrected code failed the supplied tests. "
                    "Correct the repair before rebuilding the plan.\n\n"
                    f"{execution.output}"
                )
                continue

            verification = self._verifier.verify(
                case=case,
                planner_output=planner_output,
            )

            if verification.accepted:
                return VerifiedPlan(
                    output=planner_output,
                    verification=verification,
                    attempts=attempt,
                )

            last_error = "; ".join(verification.errors)
            regeneration_feedback = (
                verification.regeneration_feedback.strip()
                or "The planner output failed oracle verification."
            )

        raise PlanGenerationError(
            f"Could not generate a verified plan for {case.case_id!r} after "
            f"{self._max_attempts} attempts. Last error: {last_error}"
        )
