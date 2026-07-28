from openai import OpenAI

from ..agents.agent import Agent
from ..common.config import AGENT_PROMPTS
from ..common.message import Message


class TutorAgent(Agent):
    llm: OpenAI
    model: str
    instructions: str

    def __init__(self, llm: OpenAI, model: str, instructions: str):
        super().__init__(llm, model, instructions)

    def get_reply(self, history: list[Message]) -> Message:
        response = self.llm.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=TutorAgent.get_prompt(history=history),
        )
        return Message(role="tutor", content=response.output_text)

    @staticmethod
    def get_prompt(history: list[Message]):
        return AGENT_PROMPTS["tutor_agent"]["dialogue_prompt"](history)
