from typing import Any

from ...common.config import AGENT_PROMPTS
from ...common.message import Message
from ..agent import Agent


class OfflineTurnVerifierAgent(Agent):
    llm: Any
    model: str
    instructions: str

    def __init__(self, llm: Any, model: str, instructions: str):
        super().__init__(llm, model, instructions)

    def get_reply(
        self, history: list[Message], tutor_responses: list[Message]
    ) -> Message:
        response = self.llm.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=OfflineTurnVerifierAgent.get_prompt(
                history=history, tutor_responses=tutor_responses
            ),
        )
        return Message(role="judge", content=response.output_text)

    @staticmethod
    def get_prompt(history: list[Message], tutor_responses: list[Message]):
        return AGENT_PROMPTS["judge_agent"]["dialogue_prompt"](history, tutor_responses)
