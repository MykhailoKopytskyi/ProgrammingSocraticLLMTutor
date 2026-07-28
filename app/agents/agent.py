from abc import ABC, abstractmethod

from openai import OpenAI

from ..common.message import Message


class Agent(ABC):
    def __init__(self, llm: OpenAI, model: str, instructions: str):
        self.llm = llm
        self.model = model
        self.instructions = instructions

    @abstractmethod
    def get_reply(self, history: list[Message]):
        pass
