from __future__ import annotations

from typing import Any

from pydantic import Field

from app.agents.agent import Agent
from app.common.config import OFFLINE_PROBLEM_TRANSLATION_AGENT_INSTRUCTIONS
from app.common.models import StrictModel


class ProblemTranslation(StrictModel):
    translated_problem_statement: str = Field(min_length=1)


class OfflineProblemTranslationAgent(Agent):
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
            instructions=OFFLINE_PROBLEM_TRANSLATION_AGENT_INSTRUCTIONS,
        )

    def translate(self, problem_statement: str, feedback: str | None = None) -> str:
        prompt = problem_statement.strip()
        if feedback:
            prompt += (
                "\n\nA previous translation was rejected by the case verifier.\n"
                f"Verifier feedback:\n{feedback}\n"
                "Regenerate the translation and specifically correct this problem."
            )
        output = self._get_structured_output(
            prompt=prompt,
            output_type=ProblemTranslation,
        )
        return output.translated_problem_statement.strip()
