from types import SimpleNamespace

from app.agents.agent import Agent
from app.common.models import StrictModel


class ExampleOutput(StrictModel):
    value: str


class FakeResponses:
    def parse(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        text_format: type[ExampleOutput],
    ) -> SimpleNamespace:
        assert model == "fake-model"
        assert instructions == "fake instructions"
        assert input == "return a structured value"
        assert text_format is ExampleOutput

        return SimpleNamespace(
            output_parsed=ExampleOutput(value="works"),
            output_text="",
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class ExampleAgent(Agent):
    def generate(self) -> ExampleOutput:
        return self._get_structured_output(
            prompt="return a structured value",
            output_type=ExampleOutput,
        )


def test_agent_returns_validated_structured_output():
    agent = ExampleAgent(
        llm=FakeClient(),
        model="fake-model",
        instructions="fake instructions",
    )

    result = agent.generate()

    assert isinstance(result, ExampleOutput)
    assert result.value == "works"
