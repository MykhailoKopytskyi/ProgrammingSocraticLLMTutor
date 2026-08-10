from abc import ABC
from typing import Any, TypeVar

from pydantic import BaseModel

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class AgentResponseError(RuntimeError):
    """Raised when an agent fails to return requested structured result"""


class Agent(ABC):
    def __init__(self, llm: Any, model: str, instructions: str):
        self.llm = llm
        self.model = model
        self.instructions = instructions

    def _get_structured_output(
        self, prompt: str, output_type: type[StructuredOutputT]
    ) -> StructuredOutputT:
        """Request structured output and validate it against a model."""

        response = self.llm.responses.parse(
            model=self.model,
            instructions=self.instructions,
            input=prompt,
            text_format=output_type,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raw_output = getattr(response, "output_text", "")
            raise AgentResponseError(
                "The model returned no parsed structured output. "
                f"Raw output: {raw_output[:500]}"
            )
        return parsed
