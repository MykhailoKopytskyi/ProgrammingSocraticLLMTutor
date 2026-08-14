from __future__ import annotations

from typing import Any

from pydantic import Field

from app.agents.agent import Agent
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
            instructions=(
                """Translate the supplied programming problem into clear English. 
                Preserve every requirement, input/output rule, example, literal 
                string, number, identifier and constraint. Do not solve the problem, 
                add requirements or include commentary. """
            ),
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
