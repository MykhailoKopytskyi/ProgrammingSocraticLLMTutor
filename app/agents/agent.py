from abc import ABC
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class AgentResponseError(RuntimeError):
    """Raised when an agent fails to return requested structured result"""


class Agent(ABC):
    def __init__(
        self,
        llm: Any,
        model: str,
        instructions: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ):
        self.llm = llm
        self.model = model
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def _get_structured_output(
        self,
        prompt: str,
        output_type: type[StructuredOutputT],
    ) -> StructuredOutputT:
        """Request structured output and validate it against a model."""

        

        request = {
            "model": self.model,
            "instructions": self.instructions,
            "input": prompt,
            "text_format": output_type,
        }

        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens

        if self.reasoning_effort:
            request["reasoning"] = {
                "effort": self.reasoning_effort,
            }

        response = self.llm.responses.parse(**request)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raw_output = getattr(response, "output_text", "")
            raise AgentResponseError(
                "The model returned no parsed structured output. "
                f"Raw output: {raw_output[:500]}"
            )
        return parsed
