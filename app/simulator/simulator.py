import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ..agents.judge_agent import JudgeAgent
from ..agents.student_agent import StudentAgent
from ..agents.tutor_agent import TutorAgent
from ..common.code_misconception import CodeMisconception
from ..common.config import AGENT_PROMPTS
from ..common.message import Message


class Simulator:
    history: list[Message]
    llm: OpenAI
    student_agent: StudentAgent
    tutor_agent: TutorAgent
    judge_agent: JudgeAgent
    misconception: CodeMisconception

    def __init__(self, misconception: CodeMisconception):
        ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(ENV_PATH)

        api_key = os.getenv("OPENAI_API_KEY")
        student_llm_model = os.getenv("STUDENT_LLM_MODEL")
        tutor_llm_model = os.getenv("TUTOR_LLM_MODEL")
        judge_llm_model = os.getenv("JUDGE_LLM_MODEL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        if not student_llm_model:
            raise ValueError("STUDENT_LLM_MODEL is not configured")

        if not tutor_llm_model:
            raise ValueError("TUTOR_LLM_MODEL is not configured")

        if not judge_llm_model:
            raise ValueError("JUDGE_LLM_MODEL is not configured")

        self.misconception = misconception
        self.llm = OpenAI(api_key=api_key)
        self.history = []
        self.student_agent = StudentAgent(
            llm=self.llm,
            model=student_llm_model,
            instructions=AGENT_PROMPTS["student_agent"]["instructions"](misconception),
            misconception=misconception,
        )
        self.tutor_agent = TutorAgent(
            llm=self.llm,
            model=tutor_llm_model,
            instructions=AGENT_PROMPTS["tutor_agent"]["instructions"],
        )
        self.judge_agent = JudgeAgent(
            llm=self.llm,
            model=judge_llm_model,
            instructions=AGENT_PROMPTS["judge_agent"]["instructions"],
        )

    # def run(self) -> SimulatorResults:
    def run(self) -> list[Message]:
        is_student_turn = True
        cur_turn_num: int = 0

        while cur_turn_num < 5:
            reply: Message
            if is_student_turn:
                reply = self.student_agent.get_reply(self.history)
            else:
                # temp_tutor_responses: list[Message] = []
                # # Generate 5 candidate Tutor agent replies
                # for _ in range(5):
                #     temp_tutor_responses.append(
                #         Message(
                #             role="tutor",
                #             content=self.tutor_agent.get_reply(self.history),
                #         )
                #     )

                reply = self.tutor_agent.get_reply(self.history)

            self.history.append(reply)
            cur_turn_num += 1
            if is_student_turn:
                is_student_turn = False
            else:
                is_student_turn = True

        return self.history
