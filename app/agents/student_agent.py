from openai import OpenAI

from ..agents.agent import Agent
from ..common.code_misconception import CodeMisconception
from ..common.config import AGENT_PROMPTS
from ..common.message import Message


class StudentAgent(Agent):
    llm: OpenAI
    model: str
    instructions: str
    misconception: CodeMisconception

    def __init__(
        self,
        llm: OpenAI,
        model: str,
        instructions: str,
        misconception: CodeMisconception,
    ):
        super().__init__(llm, model, instructions)
        self.misconception = misconception

    def get_reply(self, history: list[Message]) -> Message:
        response = self.llm.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=StudentAgent.get_prompt(history=history),
        )
        return Message(role="student", content=response.output_text)

    # def __get_instruction(self):
    #     return AGENT_PROMPTS["student_agent"]["dialogue_prompt"](self.misconception)

    @staticmethod
    def get_prompt(history: list[Message]) -> str:
        if len(history) == 0:
            return AGENT_PROMPTS["student_agent"]["initial_dialogue_prompt"]

        return AGENT_PROMPTS["student_agent"]["dialogue_prompt"](history)
