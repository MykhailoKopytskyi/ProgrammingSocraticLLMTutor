from __future__ import annotations

from typing import Any

from ...agents.agent import Agent
from ...common.config import (
    OFFLINE_DIALOGUE_VERIFIER_INSTRUCTIONS,
)
from ...common.models import BenchmarkCase, PlannerOutput, StrictModel


class OfflineDialogueVerifierAgent(Agent):
    """
    Applies the final KEEP/DROP quality check to a completed synthetic dialogue.
    """

    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str = OFFLINE_DIALOGUE_VERIFIER_INSTRUCTIONS,
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
        planner_output: PlannerOutput,
        dialogue_transcript: str,
        completion_evidence: str,
    ) -> DialogueVerification:
        if not dialogue_transcript.strip():
            raise ValueError("dialogue_transcript must not be empty")

        if not completion_evidence.strip():
            raise ValueError("completion_evidence must not be empty")

        prompt = (
            "Judge the completed tutoring dialogue.\n\n"
            "RUNTIME-VISIBLE CASE:\n"
            f"{case.visible_context()}\n\n"
            "TRAINING-ONLY ORACLE:\n"
            f"{case.oracle_context()}\n\n"
            "VERIFIED PLANNER OUTPUT:\n"
            f"{planner_output.model_dump_json(indent=2)}\n\n"
            "COMPLETED DIALOGUE:\n"
            f"{dialogue_transcript.strip()}\n\n"
            "COMPLETION EVIDENCE:\n"
            f"{completion_evidence.strip()}"
        )

        return self._get_structured_output(
            prompt=prompt,
            output_type=DialogueVerification,
        )


class DialogueVerification(StrictModel):
    accepted: bool
    main_issue: str
    errors: list[str]
    regeneration_feedback: str
