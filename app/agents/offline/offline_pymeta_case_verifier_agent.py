from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from ...common.config import OFFLINE_PYMETA_CASE_VERIFIER_INSTRUCTIONS
from ...common.models import BugAnnotation, StrictModel
from ..agent import Agent


class PyMetaCaseVerification(StrictModel):
    source_inconsistency: bool
    translation_error: bool
    repair_error: bool
    annotation_error: bool
    test_error: bool

    reasons: list[str] = Field(default_factory=list, max_length=5)
    regeneration_feedback: str | None = None

    @property
    def accepted(self) -> bool:
        return not any(
            (
                self.source_inconsistency,
                self.translation_error,
                self.repair_error,
                self.annotation_error,
                self.test_error,
            )
        )

    @property
    def regenerate_from(
        self,
    ) -> Literal[
        "translation",
        "repair",
        "annotations",
        "tests",
        "drop",
        "none",
    ]:
        if self.source_inconsistency:
            return "drop"
        if self.translation_error:
            return "translation"
        if self.repair_error:
            return "repair"
        if self.annotation_error:
            return "annotations"
        if self.test_error:
            return "tests"
        return "none"


class OfflinePyMetaCaseVerifierAgent(Agent):
    def __init__(
        self,
        llm: Any,
        model: str,
        reasoning_effort: str | None = None,
    ):
        super().__init__(
            llm=llm,
            model=model,
            reasoning_effort=reasoning_effort,
            instructions=OFFLINE_PYMETA_CASE_VERIFIER_INSTRUCTIONS,
        )

    def verify(
        self,
        *,
        original_problem: str,
        translated_problem: str,
        buggy_code: str,
        reference_code: str,
        corrected_code: str,
        bugs: list[BugAnnotation],
        tests: str,
        buggy_execution: str,
        corrected_execution: str,
        reference_execution: str,
    ) -> PyMetaCaseVerification:
        bug_data = []
        for bug in bugs:
            bug_data.append(bug.model_dump())

        prompt = f"""
ORIGINAL PYMETA PROBLEM:
{original_problem}

TRANSLATED PROBLEM:
{translated_problem}

BUGGY STUDENT CODE:
{buggy_code}

INDEPENDENT REFERENCE SOLUTION:
{reference_code}

STUDENT-ALIGNED CORRECTED CODE:
{corrected_code}

BUG ANNOTATIONS:
{json.dumps(bug_data, ensure_ascii=False, indent=2)}

GENERATED TESTS:
{tests}

BUGGY CODE EXECUTION:
{buggy_execution}

CORRECTED CODE EXECUTION:
{corrected_execution}

INDEPENDENT REFERENCE EXECUTION:
{reference_execution}
""".strip()

        return self._get_structured_output(
            prompt=prompt,
            output_type=PyMetaCaseVerification,
        )
