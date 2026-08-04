from __future__ import annotations

from dataclasses import dataclass

from ..agents.offline.offline_plan_verifier_agent import OfflinePlanVerifierAgent
from ..agents.offline.offline_planner_agent import OfflinePlannerAgent
from ..common.models import (
    BenchmarkCase,
    PedagogicalPlan,
    PlanVerification,
)


class PlanGenerationError(RuntimeError):
    """Raised when no acceptable plan is produced within the retry limit."""


@dataclass(frozen=True)
class VerifiedPlan:
    plan: PedagogicalPlan
    verification: PlanVerification
    attempts: int


def generate_verified_plan(
    *,
    case: BenchmarkCase,
    teacher_planner: OfflinePlannerAgent,
    plan_verifier: OfflinePlanVerifierAgent,
    observed_failure: str = "",
    max_attempts: int = 3,
) -> VerifiedPlan:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    regeneration_feedback = ""
    last_verification: PlanVerification | None = None

    for attempt in range(1, max_attempts + 1):
        plan = teacher_planner.generate_plan(
            case=case,
            observed_failure=observed_failure,
            regeneration_feedback=regeneration_feedback,
        )

        verification = plan_verifier.verify(
            case=case,
            plan=plan,
            observed_failure=observed_failure,
        )

        if verification.accepted:
            return VerifiedPlan(
                plan=plan,
                verification=verification,
                attempts=attempt,
            )

        last_verification = verification
        regeneration_feedback = (
            verification.regeneration_feedback.strip()
            or "The plan failed oracle verification. Correct all listed errors."
        )

    errors = last_verification.errors if last_verification is not None else []

    raise PlanGenerationError(
        f"Could not generate an accepted plan for {case.case_id!r} "
        f"after {max_attempts} attempts. Last errors: {errors}"
    )


def build_planning_record(
    *,
    case: BenchmarkCase,
    verified_plan: VerifiedPlan,
    observed_failure: str = "",
) -> dict[str, object]:
    """
    Build one neutral SFT planning record.
    Oracle information is deliberately excluded from the model input.
    """

    return {
        "task": "pedagogical_plan_generation",
        "case_id": case.case_id,
        "source": case.source,
        "input": case.visible_context(observed_failure),
        "target": verified_plan.plan.model_dump(mode="json"),
        "metadata": {
            "attempts": verified_plan.attempts,
            "covered_bug_ids": (verified_plan.verification.covered_bug_ids),
        },
    }
